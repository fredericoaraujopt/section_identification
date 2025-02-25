import os
import time
import threading
from pathlib import Path

import numpy as np
import cv2
import ipycanvas
import ipyevents
import ipywidgets as widgets
from IPython.display import display

import torch
from segment_anything import sam_model_registry, SamPredictor

##############################################################################
# Helper: create_embedding_if_needed
##############################################################################

def create_embedding_if_needed(image_path, checkpoint, model_type="vit_h", device="cuda"):
    """
    If a .npy embedding doesn't already exist in the folder of image_path, create one using SAM's
    image encoder. We'll store the result as <image_path>_embedding.npy
    so we can re-run prompts quickly.
    """
    file_directory = f"{os.path.splitext(image_path)[0]}_files"
    embedding_path = f"{file_directory}/{os.path.basename(os.path.splitext(image_path)[0])}_embedding.npy"
    if os.path.exists(embedding_path):
        print(f"[Info] Embedding already exists: {embedding_path}")
        return embedding_path

    print(f"[Info] Creating embedding for {image_path} with checkpoint {checkpoint}...")
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"[Error] Could not read image from path: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)
    embedding = predictor.get_image_embedding().cpu().numpy()
    np.save(embedding_path, embedding)
    print(f"[Info] Created embedding file: {embedding_path}")
    return embedding_path