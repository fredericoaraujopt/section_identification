# STiM — analysis & roadmap

Companion to the SAM 2.1 / CZI / ZEN upgrade. Items marked **[built]** are
implemented in this branch; the rest are **design-only** recommendations (the
beginning-to-end connectomics workflow and the low-resource/web deployment),
ordered roughly by value.

---

## 1. Section-detection weaknesses & fixes

| Weakness (before) | Impact | Fix |
|---|---|---|
| Image **compressed to ~100 KB** before SAM | Destroys resolution; section edges become mush on a 76k-px wafer montage | **[built]** retired; CZI now read from the pyramid at a chosen zoom, ordinary images at full size |
| **DBSCAN on area alone** | Debris/film/grid bars with section-like area survive; the whole-field background blob skews clusters | **[built]** shape gate (solidity, extent, aspect, drop near-full-frame/border) runs *before* area DBSCAN — see `filtering.shape_prefilter` |
| `findContours(RETR_EXTERNAL)` + no simplification | Hundreds of vertices per polygon; holes ignored; heavy for ZEN | **[built]** Douglas–Peucker (`cv2.approxPolyDP`) simplification in `export.contours_from_mask` |
| Coords exported in **compressed/overview pixels** | ROIs land in the wrong place at full res | **[built]** `CziGeometry.ds_to_full` scales every polygon back to full-res pixels |
| 16-bit brightfield fed to an RGB-trained model | Low contrast → missed sections | **[built]** `czi_io.to_rgb8` percentile-stretch + CLAHE before SAM |
| **No NMS/dedup** of overlapping masks | Same section exported twice | SAM 2.1's `box_nms_thresh` now dedupes; consider an extra IoU-based merge across crops |
| **No quality metric** (test only counts masks) | Regressions invisible | Add a small hand-labelled GT set; score detections by per-section IoU + count error (precision/recall). This is the highest-value missing piece. |

Further ideas (design-only):
- **Expected-count / grid-regularity prior.** Wafer sections sit on a near-regular lattice; fit the lattice and reject masks that don't fall on it, and *predict* missed slots.
- **Coarse-to-fine.** Detect on the overview, then re-run SAM only inside each section's bbox at full resolution to get crisp boundaries (the current single-overview pass trades boundary precision for speed).
- **Hole-aware export** (`RETR_CCOMP`) if sections legitimately contain holes.
- **MPS validation.** SAM 2's MPS path is "preliminary"; spot-check a few masks against a CPU run to rule out numerical drift.

---

## 2. Workflow efficiency

- **[built]** Never load the full 13 GB — pyramid `read(zoom=…)`; mask cache keyed by model + resolution.
- **[built]** Apple **MPS** GPU used automatically (`device.get_device`), fp32, `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- `points_per_batch` exposed as the **memory knob** (drop to 32/16 under 24 GB pressure; no effect on output).
- **Interactive iteration**: use `hiera_tiny`/`hiera_small` while tuning, `base_plus`/`large` for the final pass (swap the checkpoint; config is auto-inferred).
- **ONNX path is CPU-only** even on GPU machines (`onnx_export.py` hard-codes `CPUExecutionProvider`). Enable CUDA/CoreML providers, or retire the ONNX interactive path entirely now that editing is native in napari.
- Reuse the embedding cache (`create_embedding.py`) across sessions — already present.

---

## 3. GUI

- **[built]** Single napari window: detected sections → one **editable Shapes layer** (napari's polygon tools replace the bespoke OpenCV add/remove loop and the 16× fiducial zoom window); fiducials → a **Points layer**; SAM params exposed as widgets; a **reorderable filmstrip** for serial order.
- Next: **undo/redo** (napari has per-layer history), a real **project file** (one JSON/zarr replacing the scattered `*.pkl`/`*.npy`/`*.onnx`), parameter **presets** per objective/sample type, and a **progress callback** from SAM (long runs currently show only a busy bar).

---

## 4. Cross-correlation section reordering

**[built]** `ordering.py`: per-section normalised cross-correlation → spectral
(Fiedler) seriation, with a greedy nearest-neighbour fallback; wired to the
filmstrip and applied to exported section IDs.

Design-only improvements:
- **Shift-invariant** similarity (`skimage.registration.phase_cross_correlation`) so small misregistrations don't hurt ordering.
- **Feature-embedding** similarity (a tiny CNN / SAM features) for sections that rotate or deform between slices.
- **Confidence + manual lock**: show edge weights, let the user pin known-adjacent pairs, then re-solve.
- Recover **flip/rotation** per section (knife marks, asymmetry) — needed before EM alignment.

---

## 5. End-to-end light-microscopy → connectomics workflow (design-only)

Target pipeline, with STiM as the LM-side hub:

```
 acquire (ZEN, .czi)
   → STiM: detect sections (SAM 2.1)            [built]
   → STiM: curate polygons + fiducials          [built]
   → STiM: recover serial order (cross-corr)     [built]
   → STiM: assign IDs + export                   [built: CSV/GeoJSON/CZI]
   → ZEN Shuttle & Find: 3-pt calibration → relocate ROIs at SEM/FIB  [manual/OAD]
   → SEM array tomography acquisition
   → EM stack registration & alignment           [downstream]
```

Concrete integration targets:
- **ZEN Shuttle & Find** is the correlative bridge. ROIs/fiducials ride inside the `_STiM.czi` (this branch); the 3-point LM↔SEM calibration is an interactive per-instrument step with **no documented external file format** — the realistic automation route is **ZEN OAD** (Python/IronPython inside ZEN) operating on the live document, not authoring proprietary calibration XML offline. *Verify ROI rendering in your ZEN build first (round-trip self-test is built in).*
- **SBEMimage** — you already maintain an `sbemimage_env`. SBEMimage and Atlas/**MagC** array-tomography pipelines consume ordered section ROIs + fiducials; define a hand-off (ordered polygons + 3 fiducials + stage-µm, which the GeoJSON sidecar already carries) so STiM output drives automated acquisition. (refs: SBEMimage docs; MagC array-tomography format.)
- **Registration handshake**: emit per-section stage-µm centroids + order so the EM alignment stage starts from a known sequence.

---

## 6. Run from any low-resource laptop / on the web (design-only)

The Python/SAM 2.1 path is for power users; for "any laptop / browser" use the
**precompute-embedding + lightweight-decoder** architecture (the official SAM
web demo): the heavy encoder runs once (server GPU or a compact client-side
encoder), the small mask decoder runs in-browser via **onnxruntime-web**.

- **Encoder**: swap ViT/Hiera for **MobileSAM** (~10 MB, ~300 ms/img on ARM CPU) or **EdgeSAM** (≈40× faster than SAM) so it can run client-side; reuse the existing `onnx_export.py` decoder export and `create_embedding.py` precompute logic.
- **Runtime**: `onnxruntime-web` with **WebGPU** (encoder ≈19× / decoder ≈3.8× vs WASM) and **WASM (SIMD + threads)** fallback; cache models in OPFS. Browser support: Chrome/Edge 113+, Firefox 141+, Safari 26.
- **Packaging options**: (a) React + onnxruntime-web frontend + small **FastAPI** encoder backend (best concurrency, the SAM-demo shape); (b) fully client-side **stlite/gradio-lite** (Pyodide) for a zero-backend static page (slower cold-start, no upload of data — good for sensitive samples).
- **CZI on the web**: full CZI decode in-browser is impractical; serve a pre-rendered pyramid/overview (e.g. OME-Zarr / DeepZoom) and keep CZI writing server-side via `pylibCZIrw`.

---

## 7. Verification gaps to close

- **ZEN compatibility (highest risk).** No reference annotated CZI was available, so the injected `<Layers>` schema follows the documented Bio-Formats/ZISRAW vocabulary and is **flagged needs-verification**. Action: draw one polygon in your ZEN, save, read the XML back (`czi_export.roundtrip_check` + inspect), and mirror its exact element names/namespaces; iterate.
- **Quantitative detection QC** vs a small ground-truth set (see §1).
- **Stage-µm transform** sign/origin conventions (`CziGeometry.full_to_stage_um`) — validate against ZEN's reported stage coordinates on this file.
