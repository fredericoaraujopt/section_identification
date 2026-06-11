# STiM: Section Targeting interface for Microscopy

STiM is a napari-based interface for targeting sections on silicon wafers imaged under a microscope, developed to support large-scale connectomics workflows. The tool enables precise targeting of tissue sections for electron microscopy by retrieving their coordinates.

The interface integrates the Segment Anything Model (SAM 2.1) for automated section segmentation and includes manual-correction editors for refining results. It reads ordinary images (PNG/JPG/TIFF) and Zeiss whole-slide **CZI** files (via a downscaled pyramid overview, so a multi-GB montage is never decoded at full resolution), and exports section polygons + fiducials as CSV, GeoJSON, a high-resolution overlay PNG, and a ZEN-readable annotated CZI for the Shuttle & Find correlative workflow.

📺 **Video demo**: [STiM 🔬](https://www.loom.com/share/48c99d4387db4497963017c24cff7c3b?sid=436aefad-03f1-412b-8d9b-fa19b2b49c7f)

---

## Requirements

- **Python ≥ 3.10** (developed/tested on 3.11).
- **git** (the SAM 2 code is a git submodule — see below).
- A GPU helps but isn't required:
  - **Apple Silicon** (M-series) runs on Metal/**MPS**.
  - **Linux + NVIDIA** runs on **CUDA**.
  - **CPU-only** works but is slow; use a lighter model (`tiny`/`small`) and the low-memory option.
- Tested with: napari 0.7, PyTorch 2.x, pylibCZIrw 6.0.1, opencv-python 4.x, numpy 2.x.

---

## Installation

### 1. Clone the repo **with submodules**

SAM 2.1 lives in the `sam2/` git submodule, so clone recursively:

```bash
git clone --recursive https://github.com/fredericoaraujopt/section_identification.git
cd section_identification
```

Already cloned without `--recursive`? Pull the submodule in:

```bash
git submodule update --init --recursive
```

### 2. Create an environment

```bash
conda create -n section_identification python=3.11
conda activate section_identification
```

(or a `venv` — any Python ≥ 3.10 works).

### 3. Install PyTorch

Install the build for your platform first, so the right backend is selected:

- **macOS (Apple Silicon, MPS):** `pip install torch torchvision`
- **Linux + CUDA:** follow the selector at <https://pytorch.org/get-started/locally/> (e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`)
- **CPU-only:** `pip install torch torchvision`

### 4. Install SAM 2.1 from the submodule

Install it from `./sam2` (not PyPI). `SAM2_BUILD_CUDA=0` skips the optional CUDA extension — STiM does not need it (a harmless "cannot import name `_C`" warning is expected):

```bash
SAM2_BUILD_CUDA=0 pip install -e ./sam2
```

### 5. Install STiM and the rest of the dependencies

```bash
pip install -e .
```

This pulls napari, pylibCZIrw, opencv, scikit-learn, pycocotools, segment-anything (SAM 1 fallback for the manual editor), etc., and installs a `stim` command.

### 6. Download the SAM checkpoints

Checkpoints are large and **not** tracked in git. Put them in a `checkpoint/` folder at the repo root (the GUI's default paths):

```bash
mkdir -p checkpoint

# SAM 2.1 — automatic detection (default model)
curl -L -o checkpoint/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# SAM 1 (ViT-B) — used by the manual editor
curl -L -o checkpoint/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

For lighter/faster models on weak machines, also grab the smaller SAM 2.1 variants
(`sam2.1_hiera_tiny.pt`, `sam2.1_hiera_small.pt`, `sam2.1_hiera_large.pt`) from the
[SAM 2 release](https://github.com/facebookresearch/sam2#download-checkpoints) into the same folder; STiM's "Auto" model picks one to fit your host, or pick one in the GUI.

---

## Running STiM

Launch the GUI:

```bash
stim
# equivalently:
python -m section_identification.interface
```

Typical workflow in the GUI:

1. **Load an image** (PNG/JPG/TIFF or `.czi`). A CZI opens as a lazy full-resolution multiscale view.
2. **Calibrate** — draw 2–5 example sections in the *Calibration examples* layer. STiM sizes the section band, tile size, and SAM parameters from them and picks a model/feasible settings for your host.
3. **Run automatic detection.** SAM runs in a background process (the window stays responsive; **Stop** cancels). Detected sections appear live; the unfiltered raw layer is kept (hidden) for QC.
4. **Manually correct** with the in-viewer editor (red border = active):
   - *hover* → live mask preview under the cursor
   - **Space** → add the previewed section
   - **r** → select the section under the cursor; **r** again removes it
   - **m** → drop a fiducial
   - **d** → toggle the preview; **e** → re-embed the current view
   - click/drag pans; works at any zoom
5. **Export** the chosen formats (CSV / GeoJSON / overlay PNG / annotated CZI). Outputs are written next to the source image in `<image>_files/`; if that location isn't writable (e.g. a read-only/NTFS external drive), STiM automatically falls back to `~/STiM_exports/<image-name>/` and logs where it wrote.

Prefer code/notebooks? See `demo.ipynb` for the `automatic_identification()` → manual-edit → `export_mask_coordinates()` flow.

---

## Notes & troubleshooting

- **Apple Silicon / MPS:** STiM sets `PYTORCH_ENABLE_MPS_FALLBACK=1` for the detection worker so unsupported ops fall back to CPU. Metal uses *unified* memory shared with the OS and the napari display — if the machine is under memory pressure, close other apps or tick **low-memory (1 mask/pt)** in *Advanced*, which makes SAM emit one mask per point (~3× less peak memory) at a small recall cost. `points_per_batch` is also auto-capped to a memory-safe value per host.
- **CZI fiducials:** ZEN's "Shuttle & Find" correlative calibration markers live in the CZI metadata (stage micrometers), not as pixel annotations; read them with `section_identification.czi_io.read_shuttle_and_find_markers(path)`.
- **`cannot import name '_C' from 'sam2'`** — expected when SAM 2 is installed without the CUDA extension; results are unaffected.
- **Large local artifacts** (`*_files/`, `*.pkl`, `checkpoint/`, `images_local/`, exported CZIs/PNGs/GeoJSON) are gitignored by design.
