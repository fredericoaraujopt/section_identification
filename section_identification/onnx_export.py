import os
import sys
import subprocess
import warnings
from pathlib import Path
from PIL import Image

import torch
from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

def install_and_export_sam_onnx(
    image_path: str,
    checkpoint: str,
    model_type: str = "vit_h",
    return_single_mask: bool = True,
    opset: int = 17,
    quantize_out: bool = True,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics: bool = False,
):
    """
    Installs necessary onnx packages if missing, exports a SAM model to ONNX,
    optionally quantizes the model, and checks runtime with onnxruntime.

    Args:
        checkpoint (str): Path to the SAM .pth checkpoint file.
        output_onnx (str): Path to save the exported ONNX file.
        model_type (str): In ['vit_h','vit_l','vit_b']. Type of SAM model to load.
        return_single_mask (bool): If True, the model only returns the best mask
                                   (speeds up the pipeline for large images).
        opset (int): ONNX opset version to use in export. Must be >= 11. (Default=17)
        quantize_out (str): If True, will quantize the exported ONNX model
                            and write it to this path.
        gelu_approximate (bool): Replace GELU ops with tanh approximations (for
                                 runtimes that have slow/unimplemented erf).
        use_stability_score (bool): Replaces mask quality score with a stability score.
        return_extra_metrics (bool): If True, export includes
                                     (masks, scores, stability_scores, areas, low_res_logits).

    Returns:
        str: The path to the final ONNX model file (quantized if requested).
    """

    # ---------------------------
    # 1) Check if files already exist
    # ---------------------------
    # Define paths for the ONNX and quantized model within the directory
    file_directory = f"{os.path.splitext(image_path)[0]}_files"
    onnx_path = f"{file_directory}/{os.path.basename(os.path.splitext(image_path)[0])}_onnx.onnx"
    onnx_quantized_path = f"{file_directory}/{os.path.basename(os.path.splitext(image_path)[0])}_onnx_quantized.onnx"

    # Check if the ONNX or quantized model already exists
    if os.path.exists(onnx_quantized_path):
        print(f"Quantized ONNX model already exists: {onnx_quantized_path}")
        return onnx_quantized_path
    if os.path.exists(onnx_path):
        print(f"ONNX model already exists: {onnx_path}")
        return onnx_path

    # ---------------------------
    # 1) Install missing packages
    # ---------------------------
    required_packages = ["onnx", "onnxruntime", "onnxruntime-tools"]
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[Info] Installing missing package '{pkg}'...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

    try:
        import onnxruntime
        from onnxruntime.quantization import QuantType
        from onnxruntime.quantization.quantize import quantize_dynamic
    except ImportError:
        raise ImportError(
            "Could not import onnxruntime or quantization tools. "
            "Please verify installation or let the function install them."
        )

    # ---------------------------
    # 2) Load the SAM checkpoint
    # ---------------------------
    print("[Info] Loading SAM model from checkpoint...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    # Wrap with onnx helper
    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )

    if gelu_approximate:
        for _, module in onnx_model.named_modules():
            if isinstance(module, torch.nn.GELU):
                module.approximate = "tanh"

    # ---------------------------
    # 3) Dummy inputs for tracing
    # ---------------------------
    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size  # e.g. (64, 64) for ViT-H
    mask_input_size = [4 * x for x in embed_size]          # e.g. (256, 256)
    image = Image.open(image_path) # image dimensions

    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords":     torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels":     torch.randint(low=0, high=4,   size=(1, 5),     dtype=torch.float),
        "mask_input":       torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input":   torch.tensor([1], dtype=torch.float),
        "orig_im_size":     torch.tensor([image.height, image.width], dtype=torch.float), 
    }

    _ = onnx_model(**dummy_inputs)  # initial dry run

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
        "orig_im_size": {0: "im_size"},
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    # ---------------------------
    # 4) Export to ONNX
    # ---------------------------
    output_onnx = Path(onnx_path)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Exporting ONNX model to '{output_onnx}'...")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        with open(output_onnx, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
    print(f"[Info] ONNX export completed: {output_onnx}")

    final_onnx_path = output_onnx

    # ---------------------------
    # 5) (Optional) Quantize
    # ---------------------------
    if quantize_out:
        quantize_out = onnx_quantized_path
        print(f"[Info] Quantizing model => '{quantize_out}'...")
        quantize_dynamic(
            model_input=str(output_onnx),
            model_output=quantize_out,
            op_types_to_quantize=["MatMul", "Gemm"],
            #optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        final_onnx_path = Path(quantize_out)
        print("[Info] Quantization completed.")

    # ---------------------------
    # 6) Test with onnxruntime
    # ---------------------------
    print("[Info] Checking exported model with onnxruntime (CPU)...")
    ort_session = onnxruntime.InferenceSession(str(final_onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {k: v.detach().cpu().numpy() for k, v in dummy_inputs.items()}
    _ = ort_session.run(None, ort_inputs)
    print(f"[Success] Model '{final_onnx_path.name}' runs successfully with ONNXRuntime.")

    return str(final_onnx_path)