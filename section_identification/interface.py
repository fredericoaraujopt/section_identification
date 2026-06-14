"""STiM napari GUI.

One window: load an image (incl. whole-slide ``.czi`` from the pyramid), run SAM
2.1 detection (whole-image by default; an optional tiled mode for tiny-section
wafers), edit sections/fiducials natively in napari, and export CSV / GeoJSON /
a ZEN-annotated CZI. The working session autosaves and is restored on reopen.

Coordinate convention: napari layer data is ``(row, col)`` = ``(y, x)``; our
detection/export code uses ``(x, y)``. Helpers convert at the boundary.
"""

import ast
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QTimer, QProcess, QProcessEnvironment, QThread, Signal
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

import napari

from section_identification.section_detector import automatic_identification
from section_identification.export import export_polygons
from section_identification import czi_io
from section_identification import host_profile
from section_identification.device import describe as describe_device


def xy_to_napari(poly_xy):
    p = np.asarray(poly_xy, dtype=float).reshape(-1, 2)
    return p[:, ::-1]


def napari_to_xy(poly_yx):
    p = np.asarray(poly_yx, dtype=float).reshape(-1, 2)
    return p[:, ::-1]


# Canonical SAM 2.1 Hiera checkpoints. Each model variant is a DISTINCT network
# with its OWN weights file (the architecture is inferred from the filename in
# section_detector._infer_sam2_cfg) — "light/medium" cannot reuse the base_plus
# file. URLs/sizes from the vendored sam2/checkpoints/download_ckpts.sh.
_SAM21_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM21_CKPT_URLS = {
    "tiny": f"{_SAM21_BASE_URL}/sam2.1_hiera_tiny.pt",
    "small": f"{_SAM21_BASE_URL}/sam2.1_hiera_small.pt",
    "base_plus": f"{_SAM21_BASE_URL}/sam2.1_hiera_base_plus.pt",
    "large": f"{_SAM21_BASE_URL}/sam2.1_hiera_large.pt",
}
SAM21_CKPT_MB = {"tiny": 150, "small": 180, "base_plus": 320, "large": 900}


class _CheckpointDownloader(QThread):
    """Stream a checkpoint to ``dest`` off the UI thread, reporting % progress."""
    progress = Signal(int)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, url, dest):
        super().__init__()
        self.url, self.dest = url, dest

    def run(self):
        import urllib.request
        tmp = self.dest + ".part"
        try:
            req = urllib.request.Request(self.url, headers={"User-Agent": "STiM"})
            with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                read = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    read += len(chunk)
                    if total:
                        self.progress.emit(int(read * 100 / total))
            os.replace(tmp, self.dest)
            self.done.emit(self.dest)
        except Exception as e:  # noqa: BLE001 — surfaced to the user via the log
            try:
                os.remove(tmp)
            except OSError:
                pass
            self.failed.emit(str(e))


class _ZarrPyramidBuilder(QThread):
    """Build + persist a CZI display pyramid to Zarr off the UI thread."""
    progress = Signal(int, int)     # (level_done, level_total)
    done = Signal(str, object)      # (image_path, levels) — levels = dask-from-Zarr list
    failed = Signal(str, str)       # (image_path, message)

    def __init__(self, image_path, zpath):
        super().__init__()
        self.image_path, self.zpath = image_path, zpath

    def run(self):
        try:
            levels = czi_io.write_czi_zarr_pyramid(
                self.image_path, self.zpath,
                progress=lambda d, t: self.progress.emit(d, t))
            self.done.emit(self.image_path, levels)
        except Exception as e:  # noqa: BLE001 — surfaced to the user via the log
            # A bare KeyError repr (e.g. a dask task key) hides the real cause, so
            # send the type + full traceback to the log.
            self.failed.emit(self.image_path,
                             f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class SectionIdentificationGUI(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.image_path = None
        self.overview = None
        self.geom = None
        self.calibration = None
        self._param_viz = None
        self.image_layer = None
        self.shapes_layer = None
        self.fid_layer = None
        self.calib_layer = None
        self._zarr_builders = set()  # live _ZarrPyramidBuilder threads (GC guard)
        self.tiles_layer = None
        self.current_tile_layer = None
        self.raw_layer = None
        # autosave (debounced) + detection process + streaming state
        self._autosave_timer = QTimer(self); self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self.save_project)
        self.proc = None
        self._det_params = None
        self._det_t0 = 0.0
        self._stream_mode = False
        self._proc_buf = ""
        self._raw_sections = []
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        # Overview-px (frame) resync: the overview-px lever defines the coordinate
        # frame the worker detects in AND the display overlays masks in. When it
        # changes we re-read the overview and re-align every layer so those two
        # frames can't drift apart (debounced here; also forced before each run).
        self._frame_dirty = False
        self._frame_timer = QTimer(self); self._frame_timer.setSingleShot(True)
        self._frame_timer.timeout.connect(self._sync_overview_frame)

        layout = QVBoxLayout(); layout.setContentsMargins(6, 6, 6, 6); layout.setSpacing(4)
        self.setLayout(layout)

        # checkpoint path (set before the Advanced section uses it). SAM 2.1 is
        # used everywhere now — automatic detection AND the in-viewer editor.
        pkg = Path(os.path.abspath(__file__)); ckpt_dir = pkg.parents[1] / "checkpoint"
        self.checkpoint = str(ckpt_dir / "sam2.1_hiera_base_plus.pt")

        # collapsible section helper → returns the section's body layout
        def section(title, open=True):
            btn = QPushButton(("▾ " if open else "▸ ") + title)
            btn.setCheckable(True); btn.setChecked(open)
            btn.setStyleSheet("QPushButton{text-align:left;font-weight:bold;padding:6px;"
                              "border:none;border-radius:4px;background:#333;}")
            body = QWidget(); body.setVisible(open)
            bl = QVBoxLayout(body); bl.setContentsMargins(10, 4, 4, 8); bl.setSpacing(4)
            btn.toggled.connect(lambda on, b=body, bt=btn, t=title:
                                (b.setVisible(on), bt.setText(("▾ " if on else "▸ ") + t)))
            layout.addWidget(btn); layout.addWidget(body)
            return bl

        # ---- top (always visible): image picker + a single Help toggle ----
        top_row = QHBoxLayout()
        self.btn_select = QPushButton("Select Image / CZI…")
        self.btn_help = QPushButton("❔ Help"); self.btn_help.setFixedWidth(64)
        self.btn_help.setToolTip("How to use STiM — step-by-step guide.")
        top_row.addWidget(self.btn_select, 1); top_row.addWidget(self.btn_help)
        layout.addLayout(top_row)
        self.lbl_path = QLabel("No image selected"); self.lbl_path.setWordWrap(True)
        layout.addWidget(self.lbl_path)

        # ===== 1 · Calibrate (optional, recommended) =====
        cal = section("1 · Calibrate  (optional, recommended)", open=True)
        self.btn_calibrate = QPushButton("Calibrate from examples")
        cal.addWidget(self.btn_calibrate)
        self.lbl_calib = QLabel("Draw 1–3 example sections, then Calibrate. (See Help.)")
        self.lbl_calib.setWordWrap(True); cal.addWidget(self.lbl_calib)
        self.lbl_plan = QLabel("Detection plan: calibrate to compute it.")
        self.lbl_plan.setWordWrap(True)
        self.lbl_plan.setStyleSheet("QLabel{background:#1e1e1e;padding:6px;border-radius:4px;}")
        cal.addWidget(self.lbl_plan)

        # ===== 2 · Automatic detector =====
        det = section("2 · Automatic detector", open=True)
        host_row = QHBoxLayout()
        host_row.addWidget(QLabel("Run on:"))
        self.cb_device = QComboBox(); self.cb_device.addItems(["Auto", "CPU", "CUDA", "MPS"])
        self.cb_device.setToolTip("Where SAM runs. Auto picks CUDA > Apple MPS > CPU. "
                                  "Device sets the memory/speed regime; CPU is slowest.")
        host_row.addWidget(self.cb_device, 1)
        det.addLayout(host_row)
        self.lbl_host = QLabel(f"Host: {describe_device()}")
        self.lbl_host.setWordWrap(True); det.addWidget(self.lbl_host)
        # Live "effect" readout: how the current knobs translate into what SAM
        # actually does (section size to SAM, #tiles, est. time, resolved model +
        # whether its checkpoint is on disk). Updates as coupled knobs change.
        self.lbl_effect = QLabel("Effect: —"); self.lbl_effect.setWordWrap(True)
        self.lbl_effect.setStyleSheet("QLabel{background:#16241a;padding:6px;border-radius:4px;}")
        det.addWidget(self.lbl_effect)

        # ---- Advanced (nested fold): full SAM parameter set; each row has a
        #      tooltip + a "?" that opens that parameter's section in the guide ----
        self.btn_adv = QPushButton("▸ Advanced parameters"); self.btn_adv.setCheckable(True)
        det.addWidget(self.btn_adv)
        adv = QWidget(); adv.setVisible(False)
        # left=0 aligns the Advanced content (incl. the checkpoint button) with the
        # section's other buttons; right=10 leaves room so the "?" isn't clipped.
        advcol = QVBoxLayout(adv); advcol.setContentsMargins(0, 4, 10, 4)
        self.btn_guide = QPushButton("📖 Open parameter guide")
        advcol.addWidget(self.btn_guide)
        self.chk_viz = QCheckBox("👁 Preview parameters on the image (live)")
        self.chk_viz.setToolTip("Overlay SAM's query-point grid, the tile grid, its "
                                "sub-crops and a min-area disc — they update as you "
                                "change the values, so you see how SAM will behave.")
        advcol.addWidget(self.chk_viz)
        self.btn_adv.toggled.connect(
            lambda on: (adv.setVisible(on),
                        self.btn_adv.setText(("▾ " if on else "▸ ") + "Advanced parameters")))

        def _group(title):
            gb = QGroupBox(title)
            f = QFormLayout(gb); f.setContentsMargins(8, 6, 10, 6); f.setSpacing(4)
            f.setRowWrapPolicy(QFormLayout.WrapLongRows)
            f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            advcol.addWidget(gb)
            return f

        def _row(widget, label, anchor, tip, form):
            widget.setToolTip(tip)
            q = QPushButton("?"); q.setFixedWidth(22)
            q.setToolTip("What does this do? (opens the guide)")
            q.clicked.connect(lambda _=False, a=anchor: self._open_guide(a))
            cont = QWidget(); h = QHBoxLayout(cont); h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(widget, 1); h.addWidget(q)
            form.addRow(label, cont)
            return widget

        # ===================================================================
        # Advanced parameters, grouped by EFFECT (not pipeline stage). Each
        # cluster has ONE primary lever (★); the other knobs in it mostly move
        # the same needle and are coupled — see the guide's "Parameter clusters".
        # ===================================================================

        # 1 · Resolution — how big a section looks to SAM (SAM resizes to 1024).
        #     overview px is the ONLY knob that adds real detail; tile px / crop
        #     layers only magnify it. Calibration sets tile px + points/side from
        #     the example section, so these are normally left untouched.
        gres = _group("1 · Resolution  (how big SAM sees a section)")
        self.sp_target = _row(QSpinBox(), "★ overview px", "overview_long_side",
            "Resolution the source is read at. Calibration sets it automatically to "
            "N×1024 so SAM's tiles are 1024 px (1:1 into the encoder): 1024 = whole "
            "image, higher = N×N tiles for sections too small to resolve whole. Bigger "
            "= more detail, memory and time; host-capped. Reload to apply.", gres)
        self.sp_target.setRange(1024, 16384); self.sp_target.setSingleStep(1024); self.sp_target.setValue(1024)
        self.sp_tile = _row(QSpinBox(), "tile px  (auto · 0=whole)", "tile_px",
            "Set by calibration: whole image unless a section is too small to resolve, "
            "then tiled to magnify it. Smaller tile → bigger section to SAM, but it "
            "re-slices the SAME overview pixels — no new detail. 0 = whole image; host "
            "cap may shrink it.", gres)
        self.sp_tile.setRange(0, 16384); self.sp_tile.setSingleStep(128); self.sp_tile.setValue(0)
        self.sp_crop = _row(QSpinBox(), "crop layers  (magnify only)", "crop_n_layers",
            "SAM's built-in re-cropping inside a tile. 0=off, 1=2×2 sub-crops (~5× slower). "
            "Like tile px it only magnifies overview pixels. Calibrate sets it from section size.", gres)
        self.sp_crop.setRange(0, 3); self.sp_crop.setValue(0)

        # 2 · Coverage — how many query points land on a section.
        gcov = _group("2 · Coverage  (query-point density)")
        self.sp_pps = _row(QSpinBox(), "★ points / side", "points_per_side",
            "Density of SAM's query-point grid per tile. ~2–3 points across a section. "
            "More = finds smaller/closer objects, slower. (tile px / crop layers also "
            "change effective density.)", gcov)
        self.sp_pps.setRange(4, 192); self.sp_pps.setValue(32)
        self.sp_cropds = _row(QSpinBox(), "crop grid ÷", "crop_n_points_downscale_factor",
            "Thins the point grid on deeper crop layers. 2 is typical when crop layers ≥ 1; "
            "inert when crop layers = 0.", gcov)
        self.sp_cropds.setRange(1, 4); self.sp_cropds.setValue(1)

        # 3 · Quality gates — drop low-quality masks (the two thresholds overlap).
        gqual = _group("3 · Quality gates")
        self.sp_iou = _row(QDoubleSpinBox(), "★ pred IoU", "pred_iou_thresh",
            "SAM's confidence floor. Lower to recover faint sections; raise to drop weak ones.", gqual)
        self.sp_iou.setRange(0.0, 1.0); self.sp_iou.setSingleStep(0.05); self.sp_iou.setValue(0.80)
        self.sp_stab = _row(QDoubleSpinBox(), "stability", "stability_score_thresh",
            "Mask edge-stability floor — a second quality gate that overlaps with pred IoU. "
            "Lower for noisy/small sections; raise for clean ones.", gqual)
        self.sp_stab.setRange(0.0, 1.0); self.sp_stab.setSingleStep(0.01); self.sp_stab.setValue(0.92)
        self.sp_staboff = _row(QDoubleSpinBox(), "stability offset", "stability_score_offset",
            "Sub-knob of stability: the nudge used to measure it. Usually leave at 1.0.", gqual)
        self.sp_staboff.setRange(0.1, 5.0); self.sp_staboff.setSingleStep(0.1); self.sp_staboff.setValue(1.0)
        self.chk_m2m = _row(QCheckBox(), "refine (use_m2m)", "use_m2m",
            "Extra mask-to-mask refinement: cleaner edges, ~2× slower.", gqual)

        # 4 · Deduplication — suppress overlapping/duplicate masks.
        gdedup = _group("4 · Deduplication")
        self.sp_boxnms = _row(QDoubleSpinBox(), "★ box NMS", "box_nms_thresh",
            "Merge masks overlapping more than this. Lower = more dedup; raise to keep "
            "touching sections separate.", gdedup)
        self.sp_boxnms.setRange(0.1, 1.0); self.sp_boxnms.setSingleStep(0.05); self.sp_boxnms.setValue(0.70)
        self.sp_overlap = _row(QDoubleSpinBox(), "tile overlap", "overlap",
            "Overlap between STiM tiles so each section fits whole in ≥1 tile (edge masks "
            "are dropped to avoid duplicates).", gdedup)
        self.sp_overlap.setRange(0.0, 0.6); self.sp_overlap.setSingleStep(0.05); self.sp_overlap.setValue(0.2)
        self.sp_cropov = _row(QDoubleSpinBox(), "crop overlap", "crop_overlap_ratio",
            "Overlap between SAM's sub-crops so edge sections aren't split; inert when "
            "crop layers = 0.", gdedup)
        self.sp_cropov.setRange(0.0, 0.8); self.sp_cropov.setSingleStep(0.02); self.sp_cropov.setValue(512 / 1500)

        # 5 · Keep / drop by size — three area filters at different stages.
        gsize = _group("5 · Keep / drop by size  (3 overlapping filters)")
        self.chk_filter = _row(QCheckBox(), "★ area DBSCAN", "dbscan",
            "Keep the dominant section-sized area cluster (drops debris/clumps). "
            "Leave on for wafers.", gsize)
        self.chk_filter.setChecked(True)
        self.sp_minarea = _row(QSpinBox(), "min section area", "min_section_area",
            "Drops whole detections smaller than this + anchors the area-DBSCAN band "
            "(overview px). Calibrate sets it to ~½ the median section.", gsize)
        self.sp_minarea.setRange(0, 10_000_000); self.sp_minarea.setValue(50)
        self.sp_minmask = _row(QSpinBox(), "min mask area", "min_mask_region_area",
            "SAM-internal specks/holes filter INSIDE each mask (tile px — scales with "
            "resolution). ~5% of a section's area.", gsize)
        self.sp_minmask.setRange(0, 10_000_000); self.sp_minmask.setValue(100)

        # 6 · Performance — these never change the masks, only speed/memory.
        gperf = _group("6 · Performance  (no effect on the masks)")
        self.cb_model = _row(QComboBox(), "model", "model",
            "Backbone size = speed↔accuracy. Each variant is a SEPARATE checkpoint "
            "(tiny/small/base_plus/large); a missing one silently falls back to base_plus — "
            "see the checkpoint line below. Auto picks by host.", gperf)
        self.cb_model.addItems(["Auto", "tiny", "small", "base_plus", "large"])
        self.sp_ppb = _row(QSpinBox(), "points / batch", "points_per_batch",
            "Query points SAM runs at once. Memory/speed only — NO effect on results. "
            "Lower to avoid crashing/thrashing; raise with spare GPU. Auto-capped to host.", gperf)
        self.sp_ppb.setRange(1, 256); self.sp_ppb.setValue(16)
        self.chk_lowmem = _row(QCheckBox(), "low-memory (1 mask/pt)", "memory",
            "Memory-saver: SAM emits 1 mask per point instead of 3 → ~3× less peak mask "
            "memory (eases pressure on Macs / weak machines). May slightly lower recall on "
            "ambiguous sections. Off = SAM default (3 masks/point, best recall).", gperf)

        # crop sub-knobs are inert until crop layers ≥ 1.
        self.sp_cropov.setEnabled(False); self.sp_cropds.setEnabled(False)

        # checkpoint line: which model file is active + on-disk status, with a
        # download for the selected variant (each is a separate ~150–900 MB file).
        self.lbl_ckpt = QLabel(); self.lbl_ckpt.setWordWrap(True); advcol.addWidget(self.lbl_ckpt)
        self.btn_ckpt = QPushButton("Select checkpoint…")
        advcol.addWidget(self.btn_ckpt)
        det.addWidget(adv)                          # the Advanced fold sits in the detector section

        det_row = QHBoxLayout()
        self.btn_auto = QPushButton("Run Automatic Detection")
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setVisible(False)
        det_row.addWidget(self.btn_auto); det_row.addWidget(self.btn_stop)
        det.addLayout(det_row)
        self.lbl_elapsed = QLabel(""); det.addWidget(self.lbl_elapsed)

        # ===== 3 · Manual editor =====
        man = section("3 · Manual editor", open=False)
        self.btn_manual_napari = QPushButton("Manual editor (napari): OFF")
        self.btn_manual_napari.setCheckable(True)
        man.addWidget(self.btn_manual_napari)
        _ml = QLabel("<i>Toggle on to correct results in-place: hover + <b>Space</b> to add a "
                     "section, <b>r</b> to remove, <b>m</b> for a fiducial. See Help for all keys.</i>")
        _ml.setWordWrap(True); man.addWidget(_ml)

        # ===== 4 · Export =====
        ex = section("4 · Export", open=False)
        exp_row = QHBoxLayout()
        self.chk_exp_csv = QCheckBox("CSV"); self.chk_exp_csv.setChecked(True)
        self.chk_exp_geojson = QCheckBox("GeoJSON"); self.chk_exp_geojson.setChecked(True)
        self.chk_exp_png = QCheckBox("PNG"); self.chk_exp_png.setChecked(True)
        self.chk_exp_czi = QCheckBox("CZI"); self.chk_exp_czi.setChecked(False)
        self.chk_exp_czi.setToolTip("Annotated CZI for ZEN — copies the whole file "
                                    "(can be many GB); off by default. If the Fiducials "
                                    "layer has points, they are also written into the "
                                    "CZI's ZEN Shuttle & Find calibration markers (the "
                                    "copy only; the source is never modified).")
        for c in (self.chk_exp_csv, self.chk_exp_geojson, self.chk_exp_png, self.chk_exp_czi):
            exp_row.addWidget(c)
        ex.addLayout(exp_row)
        self.btn_export = QPushButton("Export selected")
        ex.addWidget(self.btn_export)

        # ---- bottom (always visible): log ----
        layout.addWidget(QLabel("<b>Log</b>"))
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(160)
        layout.addWidget(self.log, stretch=1)
        self.progress = QProgressBar(); self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self._old_stdout = sys.stdout
        sys.stdout = self

        self.btn_select.clicked.connect(self.select_image)
        self.btn_help.clicked.connect(self._open_help)
        self.btn_auto.clicked.connect(self.run_auto)
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_export.clicked.connect(self.export_coordinates)
        self.btn_ckpt.clicked.connect(self.select_checkpoint)
        self.btn_manual_napari.clicked.connect(self.toggle_manual_napari)
        self.btn_calibrate.clicked.connect(self.calibrate_from_examples)
        self.btn_guide.clicked.connect(lambda: self._open_guide())
        self.cb_device.currentTextChanged.connect(self._on_device_changed)
        self._device_prefer = ""
        self._refresh_host()

        # Live parameter previews: toggle + redraw whenever a geometric knob moves.
        self.chk_viz.toggled.connect(self._toggle_param_viz)
        for w in (self.sp_pps, self.sp_tile, self.sp_overlap, self.sp_crop,
                  self.sp_cropov, self.sp_cropds, self.sp_minarea, self.sp_minmask):
            w.valueChanged.connect(self._param_viz_refresh)

        # Coupling reflection: a lever updates the fields it derives + the live
        # "effect" readout. blockSignals guards (in the handlers) avoid loops.
        self._dl = None
        self.cb_model.currentTextChanged.connect(self._on_model_changed)
        self.sp_crop.valueChanged.connect(self._on_crop_layers_changed)
        for w in (self.sp_target, self.sp_tile, self.sp_pps, self.sp_overlap,
                  self.sp_minarea):
            w.valueChanged.connect(self._refresh_effect)
        self.sp_target.valueChanged.connect(self._on_overview_px_changed)
        self._update_ckpt_label()
        self._refresh_effect()
        # NB: intentionally NOT refreshing on camera move — the previews are
        # fixed in image space (a central representative tile), so they don't
        # jitter as you pan/zoom.

    def _open_guide(self, anchor=None):
        try:
            from section_identification.param_guide import open_param_guide
            open_param_guide(self, anchor)
        except Exception:
            self.log_msg("parameter guide unavailable:\n" + traceback.format_exc())

    _HELP_HTML = """
    <h2>STiM — how to use</h2>
    <ol>
      <li><b>Select Image / CZI</b> — load a wafer image or <code>.czi</code>.
          For a CZI, any existing ZEN <i>Shuttle &amp; Find</i> fiducials are
          imported onto the Fiducials layer automatically.</li>
      <li><b>Calibrate</b> (recommended) — select the <i>Calibration examples</i>
          layer, draw 1–3 example sections with napari's polygon tool, then click
          <b>Calibrate from examples</b>. STiM sizes every SAM parameter from your
          sections and picks a model that fits your machine.</li>
      <li><b>Run Automatic Detection</b> — runs SAM in the background (the window
          stays responsive; <b>Stop</b> cancels). Sections stream into the
          <i>Sections</i> layer; the parameter preview turns on so you can see the
          tiling/grid.</li>
      <li><b>Manual editor (napari)</b> — toggle <b>ON</b> (the button turns red) to
          correct results directly in the viewer:
        <ul>
          <li><b>hover</b> a section → yellow preview; <b>Space</b> adds it</li>
          <li><b>r</b> = select the section under the cursor; <b>r</b> again removes it</li>
          <li><b>m</b> = drop a fiducial; <b>d</b> = toggle the preview;
              <b>e</b> = re-embed the current view</li>
          <li>click/drag pans; works at any zoom. Toggle the button <b>OFF</b> when done.</li>
        </ul></li>
      <li><b>Fiducials</b> — CZI markers are imported as crosses. Add your own with the
          editor's <b>m</b>, or by selecting the <i>Fiducials</i> layer and using
          napari's add-point tool.</li>
      <li><b>Export</b> — tick the formats (CSV / GeoJSON / PNG / CZI) and click
          <b>Export selected</b>. Exporting a CZI with fiducials present also writes
          them into the CZI copy's ZEN Shuttle &amp; Find markers (the source is never
          modified). If the source drive is read-only, outputs go to
          <code>~/STiM_exports/</code>.</li>
    </ol>
    <p><b>Advanced parameters</b> sit under the detector, grouped by topic; each row
       has a tooltip and a <b>?</b> that opens a deeper guide. Calibrate sets the SAM
       ones for you, so you rarely need to touch them.</p>
    """

    def _open_help(self):
        """Show the step-by-step usage guide in a single reusable, non-modal dialog."""
        dlg = getattr(self, "_help_dlg", None)
        if dlg is not None:                       # reuse the existing window (don't stack)
            dlg.show(); dlg.raise_(); dlg.activateWindow(); return
        from qtpy.QtWidgets import QDialog, QTextBrowser
        dlg = QDialog(self); dlg.setWindowTitle("STiM — how to use"); dlg.resize(560, 660)
        v = QVBoxLayout(dlg)
        tb = QTextBrowser(); tb.setOpenExternalLinks(True); tb.setHtml(self._HELP_HTML)
        v.addWidget(tb)
        self._help_dlg = dlg
        dlg.show(); dlg.raise_(); dlg.activateWindow()

    def _toggle_param_viz(self, on):
        try:
            if self._param_viz is None:
                from section_identification.param_viz import ParamVisualizer
                self._param_viz = ParamVisualizer(self)
            self._param_viz.set_active(bool(on))
        except Exception:
            self.log_msg("parameter preview unavailable:\n" + traceback.format_exc())

    def _param_viz_refresh(self, *a):
        if self._param_viz is not None:
            self._param_viz.refresh_if_active()

    # ----- host profile -----
    def _current_profile(self):
        return host_profile.detect_profile(getattr(self, "_device_prefer", "") or None)

    def _on_device_changed(self, text):
        self._device_prefer = "" if text == "Auto" else text.lower()
        self._refresh_host()
        self._update_ckpt_label()      # Auto model variant may change with device
        self._refresh_effect()

    def _refresh_host(self):
        try:
            prof = self._current_profile()
            self.lbl_host.setText("Host: " + prof.summary())
            if not getattr(self, "calibration", None):
                self.sp_ppb.setValue(int(prof.points_per_batch))
        except Exception:
            pass

    def _checkpoint_for_model(self, model_pref, prof):
        """Resolve the checkpoint path for the chosen/auto model variant, falling
        back to the loaded checkpoint when the lighter variant isn't downloaded."""
        variant = prof.model_variant if model_pref in ("Auto", "") else model_pref
        d = os.path.dirname(self.checkpoint)
        cand = os.path.join(d, f"sam2.1_hiera_{variant}.pt")
        if os.path.isfile(cand):
            return cand
        if variant not in os.path.basename(self.checkpoint):
            self.log_msg(f"[host] hiera_{variant} checkpoint not found; using "
                         f"{os.path.basename(self.checkpoint)} (download "
                         f"sam2.1_hiera_{variant}.pt for the lighter/faster model).")
        return self.checkpoint

    # ----- logging -----
    def write(self, text):
        self._old_stdout.write(text); self._old_stdout.flush()
        if text.strip():
            self.log.append(text.rstrip()); QApplication.processEvents()

    def flush(self):
        pass

    def log_msg(self, text):
        print(text)

    def select_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SAM 2.1 checkpoint", "",
                                              "Checkpoints (*.pt *.pth)")
        if path:
            self.checkpoint = path
            self._update_ckpt_label()
            self._refresh_effect()

    # ----- model / checkpoint coupling -----
    def _ckpt_path_for(self, variant):
        """Path the given SAM2.1 variant's checkpoint would live at (same dir as
        the currently-loaded one)."""
        return os.path.join(os.path.dirname(self.checkpoint), f"sam2.1_hiera_{variant}.pt")

    def _resolved_variant(self):
        """(variant, on_disk, path) for the current model selection. 'Auto'
        resolves through the host profile, exactly as a run would."""
        pref = self.cb_model.currentText() if hasattr(self, "cb_model") else "Auto"
        if pref in ("Auto", ""):
            try:
                variant = self._current_profile().model_variant
            except Exception:
                variant = "base_plus"
        else:
            variant = pref
        path = self._ckpt_path_for(variant)
        on_disk = os.path.isfile(path) or (
            variant in os.path.basename(self.checkpoint) and os.path.isfile(self.checkpoint))
        return variant, on_disk, path

    def _update_ckpt_label(self):
        if not hasattr(self, "lbl_ckpt"):
            return
        variant, on_disk, _ = self._resolved_variant()
        status = "✓ on disk" if on_disk else "✗ NOT downloaded → will fall back to base_plus"
        self.lbl_ckpt.setText(
            f"Model: hiera_{variant} — {status}\nActive file: …/{os.path.basename(self.checkpoint)}")

    def _refresh_effect(self, *a):
        """Translate the current knobs into what SAM will actually do and show it
        in the live readout: section size to SAM, #tiles, est. time, resolved
        model + whether its checkpoint is present."""
        if not hasattr(self, "lbl_effect"):
            return
        try:
            variant, on_disk, _ = self._resolved_variant()
        except Exception:
            variant, on_disk = "base_plus", True
        # effective overview long side (loaded image wins; else the configured one)
        ov = getattr(self, "overview", None)
        if ov is not None:
            H, W = ov.shape[:2]
            ov_long = max(H, W)
        else:
            H = W = None
            ov_long = int(self.sp_target.value())
        tile = int(self.sp_tile.value())
        eff_tile = tile if tile > 0 else ov_long
        parts = []
        cal = getattr(self, "calibration", None)
        sec_px = float(cal["section_px"]) if (cal and cal.get("section_px")) else None
        if sec_px is not None:
            to_sam = sec_px * 1024.0 / max(eff_tile, 1)
            note = f" (+{int(self.sp_crop.value())} crop layer{'s' if self.sp_crop.value() > 1 else ''})" \
                if self.sp_crop.value() >= 1 else ""
            parts.append(f"section ≈ {to_sam:.0f}px → SAM{note}")
        # number of tiles (only meaningful once an image is loaded)
        n_tiles = None
        if H is not None:
            if eff_tile >= max(H, W):
                n_tiles = 1
            else:
                step = max(1.0, eff_tile * (1.0 - float(self.sp_overlap.value())))
                n_tiles = int(np.ceil(H / step) * np.ceil(W / step))
            parts.append(f"{n_tiles} tile{'s' if n_tiles != 1 else ''}")
            try:
                est = host_profile.estimate_run(
                    self._current_profile(), n_tiles, int(self.sp_pps.value()), variant)
                s = est["seconds"]
                parts.append(f"~{s:.0f}s" if s < 90 else f"~{s / 60:.1f} min")
            except Exception:
                pass
        parts.append(f"hiera_{variant} " + ("✓" if on_disk else "✗ falls back to base_plus"))
        self.lbl_effect.setText("Effect:  " + "  ·  ".join(parts))

    def _on_crop_layers_changed(self, v):
        """crop overlap / crop grid ÷ are inert at 0 layers — enable them only
        when crop layers ≥ 1, and bump the grid divisor to the usual 2."""
        on = int(v) >= 1
        self.sp_cropov.setEnabled(on)
        self.sp_cropds.setEnabled(on)
        if on and self.sp_cropds.value() < 2:
            self.sp_cropds.blockSignals(True); self.sp_cropds.setValue(2); self.sp_cropds.blockSignals(False)
        self._refresh_effect()

    def _on_model_changed(self, *a):
        self._update_ckpt_label()
        variant, on_disk, _ = self._resolved_variant()
        if (not on_disk) and self.cb_model.currentText() not in ("Auto", ""):
            self._maybe_offer_download(variant)
        self._refresh_effect()

    def _maybe_offer_download(self, variant):
        if getattr(self, "_dl", None) is not None and self._dl.isRunning():
            return
        mb = SAM21_CKPT_MB.get(variant, 300)
        r = QMessageBox.question(
            self, "Download model checkpoint",
            f"The hiera_{variant} checkpoint isn't on disk (~{mb} MB).\n"
            f"Without it, '{variant}' silently runs base_plus instead — no speed/size change.\n\n"
            "Download it now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if r == QMessageBox.Yes:
            self._download_checkpoint(variant)

    def _download_checkpoint(self, variant):
        url = SAM21_CKPT_URLS.get(variant)
        if not url:
            self.log_msg(f"No download URL for variant '{variant}'."); return
        dest = self._ckpt_path_for(variant)
        if os.path.isfile(dest):
            self.checkpoint = dest; self._update_ckpt_label(); self._refresh_effect()
            self.log_msg(f"{os.path.basename(dest)} already present."); return
        if getattr(self, "_dl", None) is not None and self._dl.isRunning():
            self.log_msg("A checkpoint download is already running."); return
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        self.log_msg(f"⤓ Downloading {os.path.basename(dest)} (~{SAM21_CKPT_MB.get(variant, '?')} MB)…")
        self.progress.setVisible(True); self.progress.setValue(0)
        self._dl = _CheckpointDownloader(url, dest)
        self._dl.progress.connect(self.progress.setValue)
        self._dl.done.connect(self._on_ckpt_downloaded)
        self._dl.failed.connect(self._on_ckpt_failed)
        self._dl.start()

    def _on_ckpt_downloaded(self, path):
        self.progress.setVisible(False)
        self.checkpoint = path
        self._update_ckpt_label(); self._refresh_effect()
        self.log_msg(f"✔️ Downloaded {os.path.basename(path)} — '{self.cb_model.currentText()}' is now live.")

    def _on_ckpt_failed(self, msg):
        self.progress.setVisible(False)
        self.log_msg(f"❌ Checkpoint download failed: {msg}")

    # ----- persisted Zarr display-pyramid cache -----
    def _start_zarr_build(self, image_path, zpath):
        """Kick off a background build of the on-disk Zarr pyramid for next time
        (and swap the live display to it once ready)."""
        builder = _ZarrPyramidBuilder(image_path, zpath)
        builder.progress.connect(self._on_zarr_progress)
        builder.done.connect(self._on_zarr_done)
        builder.failed.connect(self._on_zarr_failed)
        builder.finished.connect(lambda b=builder: self._zarr_builders.discard(b))
        self._zarr_builders.add(builder)
        builder.start()

    def _on_zarr_progress(self, done, total):
        self.log_msg(f"Zarr cache: level {done}/{total} written.")

    def _on_zarr_done(self, image_path, levels):
        # A different image may have been loaded while the build ran — only swap
        # the live display if it still belongs to the on-screen image.
        if image_path != self.image_path or self.image_layer is None:
            self.log_msg(f"Zarr cache built for {os.path.basename(image_path)} "
                         "(ready for next load).")
            return
        try:
            self.image_layer.data = levels   # same shapes → scale/translate unchanged
            self._display_levels = levels
            self.log_msg("✔️ Zarr cache built — display now reads the fast on-disk copy.")
        except Exception:
            self.log_msg("[warn] Zarr cache built but live swap failed (active next "
                         "load):\n" + traceback.format_exc())

    def _on_zarr_failed(self, image_path, msg):
        self.log_msg(f"[warn] Zarr cache build failed: {msg}")

    # ----- load image + restore session -----
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select an image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.czi)")
        if not path:
            return
        self.image_path = path
        self.lbl_path.setText(f"Selected: {os.path.basename(path)}")
        self.lbl_path.setToolTip(path)
        self.log_msg(f"Loading {os.path.basename(path)}…")
        self._display_scale = 1.0
        self._display_levels = None
        try:
            if czi_io.is_czi(path):
                arr, geom, meta = czi_io.read_czi_overview(
                    path, target_long_side=self.sp_target.value())
                self.geom = geom
                self.overview = czi_io.to_rgb8(arr)
                self.log_msg(f"CZI {meta['size_x']}x{meta['size_y']}, overview "
                             f"{self.overview.shape}, zoom {meta['zoom']:.4g}")
                # CZI-backed lazy multiscale for DISPLAY: napari shows a low-res
                # overview and pulls full-resolution tiles only for the visible
                # region as you zoom in. Annotation DATA stays in overview pixels
                # (so detection/save/load/export are unchanged); the Shapes layers
                # are aligned to the full-res image via a per-layer `scale`.
                try:
                    self._display_scale = 1.0 / geom.zoom  # overview px -> full-res world
                    # Prefer the persisted chunked-Zarr pyramid in the image's
                    # `_files` folder: a fast Blosc block read on every zoom/pan,
                    # instead of re-decoding the CZI region each time. Built once
                    # in the background on the first load (see _start_zarr_build).
                    from section_identification.export import resolve_export_dir
                    cache_dir = resolve_export_dir(path)
                    zpath = czi_io.zarr_pyramid_path(cache_dir, path)
                    if czi_io.zarr_pyramid_exists(zpath, path):
                        levels = czi_io.open_czi_zarr_pyramid(zpath)
                        self._display_levels = levels
                        self.log_msg(f"Cached Zarr pyramid: {len(levels)} levels "
                                     f"(L0 {levels[0].shape[1]}x{levels[0].shape[0]} px).")
                    else:
                        levels, _ = czi_io.build_czi_dask_pyramid(path)
                        self._display_levels = levels
                        self.log_msg(f"Full-res lazy multiscale: {len(levels)} levels "
                                     f"(L0 {levels[0].shape[1]}x{levels[0].shape[0]} px). "
                                     "Building on-disk Zarr cache in the background…")
                        self._start_zarr_build(path, zpath)
                except Exception:
                    self.log_msg("[warn] lazy multiscale unavailable; using overview:\n"
                                 + traceback.format_exc())
            else:
                from PIL import Image
                self.overview = np.array(Image.open(path).convert("RGB"))
                self.geom = None
        except Exception:
            self.log_msg("❌ load failed:\n" + traceback.format_exc()); return

        self._reset_layers()
        if self._display_levels is not None:
            # Full-resolution lazy pyramid (level 0 = full res); shape layers are
            # scaled by 1/zoom so overview-pixel annotations overlay correctly.
            self.image_layer = self.viewer.add_image(
                self._display_levels, name="Wafer (full-res)", multiscale=True,
                rgb=True)
        elif max(self.overview.shape[:2]) > 4096:
            pyr = [self.overview, self.overview[::2, ::2], self.overview[::4, ::4]]
            self.image_layer = self.viewer.add_image(pyr, name="Overview", multiscale=True)
        else:
            self.image_layer = self.viewer.add_image(self.overview, name="Overview")
        # Clear any parameter-preview overlays from a previously-loaded image so
        # they can't render with the old image's scale/shape on the new one.
        if getattr(self, "_param_viz", None) is not None:
            try:
                self._param_viz.set_active(False)
            except Exception:
                pass
            try:
                self.chk_viz.setChecked(False)
            except Exception:
                pass
        self._restored_raw_xy = []
        self._restored_calib_xy = []
        polys_xy, fids_xy = self._restore_session()
        try:
            self._ensure_edit_layers(polys_xy)
            if fids_xy and self.fid_layer is not None:
                self.fid_layer.data = np.asarray(fids_xy, dtype=float)[:, ::-1]
            # restore the unfiltered detector output as a (hidden) reference layer
            if self._restored_raw_xy:
                lyr = self._set_shapes(
                    "raw_layer", "Raw detections",
                    [xy_to_napari(p) for p in self._restored_raw_xy],
                    edge="orange", face=(1, 0.55, 0, 0.12), width=2)
                lyr.visible = False
                self.log_msg(f"Restored {len(self._restored_raw_xy)} raw (unfiltered) "
                             "detections (layer hidden; toggle to view).")
        except Exception:
            self.log_msg("[warn] building layers failed:\n" + traceback.format_exc())
            self._ensure_edit_layers([])
        self._raw_sections = []
        self._ensure_calib_layer()
        # For a CZI, auto-scan its ZEN Shuttle & Find calibration markers and place
        # them on the Fiducials layer (idempotent across reloads via dedup).
        if czi_io.is_czi(self.image_path):
            self.import_czi_fiducials(auto=True)
        self._frame_dirty = False    # geom now matches the current overview px
        self._refresh_effect()   # now that the overview is loaded: tiles + time estimate

    def _restore_session(self):
        """Return (polys_overview, fids_overview) from project JSON, else CZI
        annotations, else the legacy mask_coordinates.csv."""
        polys, fids = self.load_project()
        if polys:
            self.log_msg(f"Restored {len(polys)} sections + {len(fids)} fiducials "
                         "from autosaved project.")
            return polys, fids
        if czi_io.is_czi(self.image_path) and self.geom is not None:
            try:
                from section_identification.czi_export import read_annotations
                pf, ff = read_annotations(self.image_path)
                polys = [self._to_overview(p) for p in pf]
                fids = [tuple(self._to_overview([f])[0]) for f in ff]
                if polys or fids:
                    self.log_msg(f"Loaded {len(polys)} polygons + {len(fids)} "
                                 "fiducials from CZI annotations.")
                    return polys, fids
            except Exception:
                self.log_msg("[warn] CZI annotation read failed:\n" + traceback.format_exc())
        polys, fids = self._load_legacy_csv()
        if polys:
            self.log_msg(f"Restored {len(polys)} sections from legacy "
                         "mask_coordinates.csv.")
        return polys, fids

    def _reset_layers(self):
        for lyr in list(self.viewer.layers):
            try:
                self.viewer.layers.remove(lyr)
            except Exception:
                pass
        self.image_layer = self.shapes_layer = self.fid_layer = None
        self.calib_layer = self.tiles_layer = None
        self.current_tile_layer = self.raw_layer = None

    def _layer_scale(self):
        """Per-layer scale so overview-pixel annotation DATA overlays the
        full-res multiscale image (1/geom.zoom). (1,1) when not in full-res
        display mode, so ordinary images are unaffected."""
        s = getattr(self, "_display_scale", 1.0)
        return (s, s)

    def _on_overview_px_changed(self, *a):
        """The overview-px lever changed. Mark the frame stale and (debounced)
        re-read the overview so the display and the detection worker can't run in
        different coordinate frames. Inert for non-CZI (overview px is a no-op
        there — the worker reads the file at full res)."""
        if self.image_path and czi_io.is_czi(self.image_path) and self.geom is not None:
            self._frame_dirty = True
            self._frame_timer.start(500)

    def _sync_overview_frame(self):
        """Re-read the CZI overview at the CURRENT overview px and re-align every
        layer to it, so masks/tiles/preview and the detection worker share ONE
        coordinate frame. Existing annotations are migrated by the zoom ratio (so
        they keep their on-screen position) and calibrated area thresholds
        (overview px^2) are rescaled. No-op for non-CZI or when nothing changed.
        Cheap: reads a downscaled pyramid level, never the full-res image."""
        self._frame_dirty = False
        if not (self.image_path and czi_io.is_czi(self.image_path)
                and self.geom is not None):
            return
        # Don't reshape the frame out from under a streaming run.
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self._frame_dirty = True
            return
        target = int(self.sp_target.value())
        old_zoom = float(self.geom.zoom)
        try:
            arr, new_geom, _ = czi_io.read_czi_overview(
                self.image_path, target_long_side=target)
        except Exception:
            self.log_msg("[warn] overview re-read failed; frame unchanged:\n"
                         + traceback.format_exc())
            return
        new_zoom = float(new_geom.zoom)
        if abs(new_zoom - old_zoom) <= 1e-9 * max(1.0, old_zoom):
            return                                  # already in this frame
        r = new_zoom / old_zoom                     # old overview px -> new overview px
        self.overview = czi_io.to_rgb8(arr)
        self.geom = new_geom
        self._display_scale = (1.0 / new_zoom) if self._display_levels is not None else 1.0
        # The lazy full-res pyramid is frame-independent (always level0 = full res),
        # so the image layer is untouched there; only the simple overview-as-image
        # fallback shows self.overview directly and must be refreshed.
        if self._display_levels is None and self.image_layer is not None:
            try:
                if max(self.overview.shape[:2]) > 4096:
                    self.image_layer.data = [self.overview, self.overview[::2, ::2],
                                             self.overview[::4, ::4]]
                else:
                    self.image_layer.data = self.overview
            except Exception:
                pass
        # Migrate live annotation coordinates so they stay on the same features,
        # then re-apply the per-layer scale for the new frame (world position is
        # preserved: data*r * 1/new_zoom == data * 1/old_zoom).
        for attr in ("shapes_layer", "raw_layer", "calib_layer", "fid_layer"):
            lyr = getattr(self, attr, None)
            if lyr is None or lyr not in self.viewer.layers:
                continue
            try:
                d = lyr.data
                if isinstance(d, np.ndarray):
                    if d.size:
                        lyr.data = d.astype(float) * r
                elif d:
                    lyr.data = [np.asarray(p, dtype=float) * r for p in d]
            except Exception:
                pass
            try:
                lyr.scale = self._layer_scale()
            except Exception:
                pass
        # Calibrated thresholds are in overview pixels of the OLD frame.
        cal = getattr(self, "calibration", None)
        if cal:
            for k in ("min_area", "max_area"):
                if cal.get(k) is not None:
                    try:
                        cal[k] = float(cal[k]) * r * r
                    except Exception:
                        pass
            if cal.get("section_px") is not None:
                try:
                    cal["section_px"] = float(cal["section_px"]) * r
                except Exception:
                    pass
        self._param_viz_refresh()
        self._refresh_effect()
        self.log_msg(f"Overview frame re-read at {max(self.overview.shape[:2])}px "
                     f"(zoom {new_zoom:.4g}); masks/tiles/preview now overlay the "
                     "full-res image in one frame.")

    def _ensure_edit_layers(self, polygons_xy):
        if self.shapes_layer is not None and self.shapes_layer in self.viewer.layers:
            self.viewer.layers.remove(self.shapes_layer)
        data = [xy_to_napari(p) for p in polygons_xy] if polygons_xy else []
        self.shapes_layer = self.viewer.add_shapes(
            data, shape_type="polygon", name="Sections",
            face_color=[1, 0, 0, 0.18], edge_width=4, scale=self._layer_scale())
        try:
            self.shapes_layer.edge_color = "red"
        except Exception:
            pass
        if self.fid_layer is None or self.fid_layer not in self.viewer.layers:
            self.fid_layer = self.viewer.add_points(np.empty((0, 2)),
                                                    name="Fiducials", size=28,
                                                    symbol="cross",
                                                    scale=self._layer_scale())
            for attr, val in (("symbol", "cross"), ("face_color", "cyan"),
                              ("border_color", "cyan"), ("edge_color", "cyan")):
                try:
                    setattr(self.fid_layer, attr, val)
                except Exception:
                    pass
        for lyr in (self.shapes_layer, self.fid_layer):
            try:
                lyr.events.data.connect(self._schedule_autosave)
            except Exception:
                pass

    def _ensure_calib_layer(self):
        if self.calib_layer is None or self.calib_layer not in self.viewer.layers:
            data = [xy_to_napari(p) for p in getattr(self, "_restored_calib_xy", [])]
            self.calib_layer = self.viewer.add_shapes(
                data, shape_type="polygon", name="Calibration examples",
                face_color=[0, 1, 0, 0.25], edge_width=4, scale=self._layer_scale())
            try:
                self.calib_layer.edge_color = "lime"
            except Exception:
                pass
            try:                                   # persist drawn examples too
                self.calib_layer.events.data.connect(self._schedule_autosave)
            except Exception:
                pass
        return self.calib_layer

    def current_calib_xy(self):
        lyr = getattr(self, "calib_layer", None)
        if lyr is None or lyr not in self.viewer.layers:
            return []
        return [napari_to_xy(d) for d in lyr.data if len(np.asarray(d)) >= 3]

    # ----- project autosave / restore -----
    def _schedule_autosave(self, *a):
        try:
            self._autosave_timer.start(1500)
        except Exception:
            pass

    def _project_path(self):
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        return os.path.join(f"{os.path.splitext(self.image_path)[0]}_files",
                            f"{base}_stim_project.json")

    def _to_full(self, pts):
        p = np.asarray(pts, dtype=float).reshape(-1, 2)
        if self.geom is None:
            return [[float(x), float(y)] for x, y in p]
        fx, fy = self.geom.ds_to_full(p[:, 0], p[:, 1])
        return [[float(a), float(b)] for a, b in zip(fx, fy)]

    def _to_overview(self, pts):
        p = np.asarray(pts, dtype=float).reshape(-1, 2)
        if self.geom is None:
            return p
        x, y = self.geom.full_to_ds(p[:, 0], p[:, 1])
        return np.column_stack([x, y])

    def current_polygons_xy(self):
        if self.shapes_layer is None:
            return []
        return [napari_to_xy(d) for d in self.shapes_layer.data
                if len(np.asarray(d)) >= 3]

    def current_fiducials_xy(self):
        if self.fid_layer is None or len(self.fid_layer.data) == 0:
            return []
        return [tuple(map(float, napari_to_xy(p).ravel())) for p in self.fid_layer.data]

    def current_raw_xy(self):
        """All UNFILTERED detections (the 'Raw detections' layer), overview px."""
        lyr = getattr(self, "raw_layer", None)
        if lyr is None or lyr not in self.viewer.layers:
            return []
        return [napari_to_xy(d) for d in lyr.data if len(np.asarray(d)) >= 3]

    def save_project(self):
        if self.image_path is None:
            return
        try:
            data = {"image": self.image_path,
                    "sections": [self._to_full(p) for p in self.current_polygons_xy()],
                    "fiducials": [self._to_full([f])[0] for f in self.current_fiducials_xy()],
                    # full unfiltered detector output, kept for re-filtering / QC
                    "raw_sections": [self._to_full(p) for p in self.current_raw_xy()],
                    "calibration_examples": [self._to_full(p) for p in self.current_calib_xy()]}
            path = self._project_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def load_project(self):
        if self.image_path is None:
            return [], []
        path = self._project_path()
        if not os.path.isfile(path):
            return [], []
        try:
            data = json.load(open(path))
        except Exception:
            return [], []
        polys = [self._to_overview(s) for s in data.get("sections", [])]
        fids = [tuple(self._to_overview([f])[0]) for f in data.get("fiducials", [])]
        self._restored_raw_xy = [self._to_overview(s) for s in data.get("raw_sections", [])]
        self._restored_calib_xy = [self._to_overview(s) for s in data.get("calibration_examples", [])]
        return polys, fids

    def _load_legacy_csv(self):
        """Load sections from the original ``*_mask_coordinates.csv`` (overview
        coords assumed = image coords, i.e. for non-CZI images)."""
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        path = os.path.join(f"{os.path.splitext(self.image_path)[0]}_files",
                            f"{base}_mask_coordinates.csv")
        if not os.path.isfile(path):
            return [], []
        polys, fids = [], []
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    rtype = (row.get("type") or "").lower()
                    coords = row.get("contour_coordinates") or ""
                    if rtype == "fiducials":
                        try:
                            fids = [tuple(map(float, p)) for p in ast.literal_eval(coords)]
                        except Exception:
                            pass
                        continue
                    try:
                        contours = ast.literal_eval(coords)  # [[ [x,y],... ], ...]
                        cnt = max(contours, key=len)
                        poly = np.asarray(cnt, dtype=float).reshape(-1, 2)
                        if len(poly) >= 3:
                            polys.append(poly)
                    except Exception:
                        continue
        except Exception:
            return [], []
        return polys, fids

    # ----- detection (separate process) -----
    def run_auto(self):
        if self.overview is None:
            self.log_msg("⚠️ Select an image first."); return
        if not os.path.isfile(self.checkpoint):
            QMessageBox.information(self, "Missing checkpoint",
                                    f"SAM 2.1 checkpoint not found:\n{self.checkpoint}")
            self.select_checkpoint()
            if not os.path.isfile(self.checkpoint):
                return
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.log_msg("Detection already running — press Stop first."); return

        # Guarantee the display frame matches the overview px the worker is about to
        # read, so the streamed tiles/masks overlay the full-res image instead of a
        # stale-scale ghost (the worker re-reads the overview at this same value).
        if self._frame_dirty:
            self._sync_overview_frame()

        # One streaming engine (SAM's whole-image generator can't stream, so we
        # always tile — often a single whole-image tile). Every SAM parameter
        # comes from the (calibrated) Advanced fields; the host profile picks the
        # model + caps the tile so the run stays feasible on this machine.
        prof = self._current_profile()
        cal = self.calibration or {}
        tile_px = int(self.sp_tile.value())
        if tile_px <= 0:
            tile_px = max(self.overview.shape[:2])          # whole image
        tile_px = int(min(tile_px, prof.tile_cap_px))       # memory cap may force tiling
        min_area = float(cal.get("min_area", self.sp_minarea.value() or 50))
        max_area = float(cal.get("max_area", 1e12))
        ckpt = self._checkpoint_for_model(self.cb_model.currentText(), prof)
        args = ["-m", "section_identification.detect_worker",
                "--image", self.image_path, "--checkpoint", ckpt,
                "--device", getattr(self, "_device_prefer", "") or "",
                "--target-long-side", str(self.sp_target.value()),
                "--points-per-side", str(self.sp_pps.value()),
                "--points-per-batch", str(self.sp_ppb.value()),
                "--pred-iou-thresh", str(self.sp_iou.value()),
                "--stability-score-thresh", str(self.sp_stab.value()),
                "--stability-score-offset", str(self.sp_staboff.value()),
                "--box-nms-thresh", str(self.sp_boxnms.value()),
                "--crop-n-layers", str(self.sp_crop.value()),
                "--crop-overlap-ratio", str(self.sp_cropov.value()),
                "--crop-n-points-downscale-factor", str(self.sp_cropds.value()),
                "--min-mask-region-area", str(self.sp_minmask.value()),
                "--use-m2m", "1" if self.chk_m2m.isChecked() else "0",
                "--multimask", "0" if self.chk_lowmem.isChecked() else "1",
                "--tile-px", str(tile_px), "--overlap", str(self.sp_overlap.value()),
                "--min-area", str(min_area), "--max-area", str(max_area)]
        self._stream_mode = True
        self._reset_stream_layers(); self._raw_sections = []; self._det_params = None
        # Show the parameter preview (tile grid / point grid) while detecting.
        if not self.chk_viz.isChecked():
            self.chk_viz.setChecked(True)
        whole = tile_px >= max(self.overview.shape[:2])
        self.log_msg(f"▶ Detection on {prof.device} ({os.path.basename(ckpt)}): "
                     f"overview {self.sp_target.value()}px · "
                     f"{'whole image' if whole else 'tiles'}, tile_px={tile_px}, grid "
                     f"{self.sp_pps.value()}, area {min_area:.0f}–{max_area:.0f}.")
        self._proc_buf = ""

        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.proc.setProcessEnvironment(env)
        # Run the worker from a neutral cwd: the in-repo sam2/ dir would otherwise
        # shadow the installed `sam2` package when cwd == repo root.
        self.proc.setWorkingDirectory(os.path.expanduser("~"))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.finished.connect(self._on_proc_finished)
        self.proc.errorOccurred.connect(lambda e: self.log_msg(f"❌ process error: {e}"))
        self.btn_auto.setEnabled(False); self.btn_stop.setVisible(True)
        self.progress.setRange(0, 0); self.progress.setVisible(True)
        self._det_t0 = time.time(); self._elapsed_timer.start(1000)
        self.log_msg("(running in a background process — GUI stays responsive; Stop to cancel)")
        self.proc.start(sys.executable, args)

    def _tick_elapsed(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.lbl_elapsed.setText(f"⏱ {int(time.time() - self._det_t0)} s elapsed")

    def _on_proc_output(self):
        try:
            text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        except Exception:
            return
        self._proc_buf += text
        *lines, self._proc_buf = self._proc_buf.split("\n")
        for line in lines:
            line = line.rstrip()
            if not line:
                continue
            if (line.startswith("STIM_TILES ") or line.startswith("STIM_TILE ")
                    or line.startswith("STIM_TILESTART ")):
                try:
                    self._handle_stim_line(line)
                except Exception:
                    pass
            else:
                self.log_msg(line)

    def _on_proc_finished(self, code, status):
        self._elapsed_timer.stop(); self.lbl_elapsed.setText("")
        self.btn_auto.setEnabled(True); self.btn_stop.setVisible(False)
        self.progress.setVisible(False)
        self.proc = None
        if code != 0:
            self.log_msg(f"⏹ detection stopped/failed (exit {code}); partials kept.")
            self._finalize_tiled()
            self._clear_detector_overlays()
            return
        try:
            self._finalize_tiled()
        except Exception:
            self.log_msg("❌ loading results failed:\n" + traceback.format_exc())
        self._clear_detector_overlays()

    def _clear_detector_overlays(self):
        """Once a run ends, remove the detector's transient overlays — the tile
        grid, current tile, and the live parameter previews (grid/crops/min-area).
        Keep 'Raw detections' (the result) and 'Sections'."""
        for attr in ("tiles_layer", "current_tile_layer"):
            lyr = getattr(self, attr, None)
            try:
                if lyr is not None and lyr in self.viewer.layers:
                    self.viewer.layers.remove(lyr)
            except Exception:
                pass
            setattr(self, attr, None)
        # keep Raw detections (the result) but hide it by default — the kept
        # 'Sections' layer is what the user works with.
        if getattr(self, "raw_layer", None) is not None:
            try:
                self.raw_layer.visible = False
            except Exception:
                pass
        if getattr(self, "_param_viz", None) is not None:
            try:
                self._param_viz.set_active(False)
            except Exception:
                pass
        try:
            self.chk_viz.setChecked(False)
        except Exception:
            pass

    def stop_detection(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.log_msg("■ Stopping…"); self.proc.kill()

    # ----- tiled streaming display -----
    def _set_shapes(self, attr, name, data, edge="white", face=(0, 0, 0, 0), width=3):
        lyr = getattr(self, attr, None)
        if lyr is not None and lyr in self.viewer.layers:
            self.viewer.layers.remove(lyr)
        lyr = self.viewer.add_shapes(data, shape_type="polygon", name=name,
                                     face_color=list(face), edge_width=width,
                                     scale=self._layer_scale())
        try:
            lyr.edge_color = edge
        except Exception:
            pass
        setattr(self, attr, lyr)
        return lyr

    @staticmethod
    def _box_rect(box):
        x, y, w, h = box
        return np.array([[y, x], [y, x + w], [y + h, x + w], [y + h, x]], dtype=float)

    def _reset_stream_layers(self):
        for attr in ("tiles_layer", "current_tile_layer", "raw_layer"):
            lyr = getattr(self, attr, None)
            if lyr is not None and lyr in self.viewer.layers:
                try:
                    self.viewer.layers.remove(lyr)
                except Exception:
                    pass
            setattr(self, attr, None)

    def _handle_stim_line(self, line):
        if line.startswith("STIM_TILESTART "):
            d = json.loads(line[len("STIM_TILESTART "):])
            # Show the tile being worked on NOW (before SAM runs on it).
            self._set_shapes("current_tile_layer", "Current tile",
                             [self._box_rect(d["box"])],
                             edge="cyan", face=(0, 1, 1, 0.12), width=4)
            self.log_msg(f"  tile {d['k']}/{d['n']} — segmenting…")
            return
        if line.startswith("STIM_TILES "):
            boxes = json.loads(line[len("STIM_TILES "):])
            self._set_shapes("tiles_layer", "Tiles", [self._box_rect(b) for b in boxes],
                             edge="yellow", face=(1, 1, 0, 0.06), width=2)
            self.log_msg(f"Tiling into {len(boxes)} tiles…")
            return
        d = json.loads(line[len("STIM_TILE "):])
        k, n = d["k"], d["n"]
        # "Current tile" was already drawn by STIM_TILESTART; here we just stream
        # the sections this tile confirmed.
        new = [xy_to_napari(np.asarray(s["poly"], dtype=float)) for s in d["sections"]
               if len(s["poly"]) >= 3]
        for s in d["sections"]:
            if len(s["poly"]) >= 3:
                self._raw_sections.append(s)
        if new:
            if self.raw_layer is None or self.raw_layer not in self.viewer.layers:
                self._set_shapes("raw_layer", "Raw detections", new,
                                 edge="orange", face=(1, 0.55, 0, 0.25), width=3)
            else:
                try:
                    self.raw_layer.add(new, shape_type="polygon")
                except Exception:
                    self.raw_layer.data = list(self.raw_layer.data) + new
        elapsed = max(1e-3, time.time() - self._det_t0)
        eta = elapsed / k * (n - k)
        self.log_msg(f"  tile {k}/{n} · +{len(d['sections'])} (total "
                     f"{len(self._raw_sections)}) · ~{int(eta)}s left")

    def _finalize_tiled(self):
        # `raw` = ALL SAM output (debris included) — kept in the Raw layer for QC.
        # Sections = raw → size band → DBSCAN.
        raw = self._raw_sections
        cal = self.calibration or {}
        lo = float(cal.get("min_area", 0.0))
        hi = float(cal.get("max_area", float("inf")))
        band = [s for s in raw if lo <= float(s["area"]) <= hi]
        kept = band
        if self.chk_filter.isChecked() and len(band) >= 3:
            try:
                from section_identification.filtering import filtering
                ml = [{"area": float(s["area"])} for s in band]
                alo = max(1.0, min(m["area"] for m in ml)); ahi = max(m["area"] for m in ml) + 1
                chosen, _ = filtering(ml, np.linspace(alo, ahi, 12), range(2, 5))
                ids = {id(m) for m in chosen}
                kept = [s for s, m in zip(band, ml) if id(m) in ids]
            except Exception:
                kept = band
        self._ensure_edit_layers([np.asarray(s["poly"], dtype=float) for s in kept])
        self.log_msg(f"✔️ {len(raw)} raw (all SAM, incl. debris) → {len(band)} in "
                     f"size band [{lo:.0f}–{hi:.0f}] → {len(kept)} kept → 'Sections'. "
                     "Raw layer keeps everything (hidden; toggle for QC).")
        self.save_project()

    # ----- calibration + preview -----
    def calibrate_from_examples(self):
        if self.overview is None:
            self.log_msg("⚠️ Load an image first."); return
        lyr = self._ensure_calib_layer()
        polys = [napari_to_xy(d) for d in lyr.data if len(np.asarray(d)) >= 3]
        if not polys:
            self.log_msg("Draw 2–5 example sections in 'Calibration examples' first.")
            try:
                self.viewer.layers.selection.active = lyr
            except Exception:
                pass
            return
        from section_identification.calibration import calibrate, summary
        try:
            prof = self._current_profile()
            cal = calibrate(polys, geom=self.geom,
                            overview_long_side=max(self.overview.shape[:2]), profile=prof)
        except Exception:
            self.log_msg("❌ calibration failed:\n" + traceback.format_exc()); return

        # Encoder-native plan: read the overview at the recommended N·1024 so SAM's
        # tiles are 1024 px (1:1 into the encoder). Reload there — UP when sections are
        # too small to resolve whole, DOWN when a needlessly fine overview is just being
        # re-tiled for memory — then re-measure the migrated examples so the plan is
        # consistent in the frame detection will actually run in.
        rec = cal.get("recommended_overview_long_side")
        cur = max(self.overview.shape[:2])
        if rec and rec != cur and czi_io.is_czi(self.image_path) and self.geom is not None:
            self.sp_target.setValue(int(rec))
            self._sync_overview_frame()           # re-read at rec + migrate examples now
            polys = [napari_to_xy(d) for d in self._ensure_calib_layer().data
                     if len(np.asarray(d)) >= 3]
            if polys:
                try:
                    cal = calibrate(polys, geom=self.geom,
                                    overview_long_side=max(self.overview.shape[:2]), profile=prof)
                except Exception:
                    self.log_msg("❌ re-calibration failed:\n" + traceback.format_exc())
            n = max(1, round(rec / 1024))
            self.log_msg(f"Overview set to {rec}px so SAM tiles are 1024 px" +
                         (" — whole image." if n <= 1 else f" — {n}×{n} tiles."))

        self.calibration = cal
        self._apply_calibration_to_ui(cal, prof)
        self.log_msg("✔️ " + summary(cal))
        self.log_msg("→ " + cal.get("plan_summary", ""))

    def _apply_calibration_to_ui(self, cal, prof=None):
        """Write the calibrated SAM parameters into the Advanced fields + plan."""
        def setv(widget, key, cast):
            if key in cal:
                try:
                    widget.setValue(cast(cal[key]))
                except Exception:
                    pass
        setv(self.sp_pps, "points_per_side", int)
        setv(self.sp_iou, "pred_iou_thresh", float)
        setv(self.sp_stab, "stability_score_thresh", float)
        setv(self.sp_staboff, "stability_score_offset", float)
        setv(self.sp_boxnms, "box_nms_thresh", float)
        setv(self.sp_crop, "crop_n_layers", int)
        setv(self.sp_cropov, "crop_overlap_ratio", float)
        setv(self.sp_cropds, "crop_n_points_downscale_factor", int)
        setv(self.sp_minmask, "min_mask_region_area", int)
        setv(self.sp_minarea, "min_area", int)          # DBSCAN area floor
        setv(self.sp_overlap, "overlap", float)
        # tile_px: 0 (whole image) unless tiling is recommended
        self.sp_tile.setValue(int(cal.get("tile_px", 0)) if cal.get("tiling_recommended") else 0)
        if prof is not None:
            self.sp_ppb.setValue(int(prof.points_per_batch))
            self.lbl_host.setText("Host: " + prof.summary())
        self.lbl_plan.setText("Plan: " + cal.get("plan_summary", ""))
        self._on_crop_layers_changed(self.sp_crop.value())   # sync crop sub-knob state
        self._update_ckpt_label()
        self._refresh_effect()

    def preview_tiling(self):
        if self.overview is None:
            self.log_msg("⚠️ Load an image first."); return
        from section_identification.tiled_detect import plan_tiles
        H, W = self.overview.shape[:2]
        tile_px = int(self.calibration["tile_px"]) if self.calibration else 768
        overlap = float(self.calibration.get("overlap", 0.25)) if self.calibration else 0.25
        boxes = plan_tiles(W, H, tile_px, overlap)
        self._set_shapes("tiles_layer", "Tiles (preview)",
                         [self._box_rect(b) for b in boxes], edge="yellow",
                         face=(1, 1, 0, 0.10), width=2)
        self.log_msg(f"Preview: {len(boxes)} tiles of {tile_px}px (overlap {overlap}); "
                     f"SAM upscale ×{1024.0 / tile_px:.1f}.")

    # ----- manual (napari in-viewer) editor -----
    def _style_manual_btn(self, active):
        """Unmistakable ON/OFF state for the manual-editor toggle (text + colour)."""
        self.btn_manual_napari.setChecked(bool(active))
        if active:
            self.btn_manual_napari.setText("● Manual editor: ON — click to stop")
            self.btn_manual_napari.setStyleSheet(
                "QPushButton{background:#b23b3b;color:white;font-weight:bold;padding:6px;"
                "border-radius:4px;}")
        else:
            self.btn_manual_napari.setText("Manual editor (napari): OFF")
            self.btn_manual_napari.setStyleSheet("")

    def toggle_manual_napari(self):
        """Activate/deactivate the in-viewer (napari) SAM editor."""
        if self.overview is None:
            self.log_msg("⚠️ Load an image first.")
            self._style_manual_btn(False)
            return
        try:
            if getattr(self, "_napari_editor", None) is None:
                from section_identification.napari_sam_editor import NapariSamEditor
                self._napari_editor = NapariSamEditor(self)
            active = self._napari_editor.toggle()
            self._style_manual_btn(active)
        except Exception:
            self._style_manual_btn(False)
            self.log_msg("❌ napari editor error:\n" + traceback.format_exc())

    # ----- export -----
    def export_coordinates(self):
        if self.image_path is None:
            self.log_msg("⚠️ Nothing to export."); return
        polys = self.current_polygons_xy()
        if not polys:
            self.log_msg("⚠️ No section polygons to export."); return
        section_ids = [f"section_{k}" for k in range(1, len(polys) + 1)]
        fids = self.current_fiducials_xy()
        fmts = dict(write_csv=self.chk_exp_csv.isChecked(),
                    write_geojson=self.chk_exp_geojson.isChecked(),
                    write_png=self.chk_exp_png.isChecked(),
                    write_czi=self.chk_exp_czi.isChecked())
        if not any(fmts.values()):
            self.log_msg("⚠️ Select at least one export format (CSV/GeoJSON/PNG/CZI)."); return
        # Auto: writing an annotated CZI + having fiducials => also write them into
        # the CZI's ZEN Shuttle & Find calibration markers (no separate toggle).
        fmts["write_sf"] = bool(fmts["write_czi"] and czi_io.is_czi(self.image_path) and fids)
        chosen = ", ".join({"write_csv": "CSV", "write_geojson": "GeoJSON",
                            "write_png": "PNG", "write_czi": "CZI",
                            "write_sf": "S&F markers"}[k] for k, v in fmts.items() if v)
        self.log_msg(f"▶ Exporting {len(polys)} sections, {len(fids)} fiducials → {chosen}…")
        try:
            outputs = export_polygons(self.image_path, polys, fids, geom=self.geom,
                                      section_ids=section_ids, **fmts)
            out_dir = outputs.get("dir")
            default_dir = f"{os.path.splitext(self.image_path)[0]}_files"
            if out_dir and os.path.abspath(out_dir) != os.path.abspath(default_dir):
                self.log_msg(f"ℹ️ Source folder isn't writable (read-only drive?) — "
                             f"exported to {out_dir} instead.")
            files = {k: v for k, v in outputs.items() if k != "dir"}
            self.log_msg("✔️ Exported: " + ", ".join(f"{k}={v}" for k, v in files.items()))
        except Exception:
            self.log_msg("❌ export error:\n" + traceback.format_exc())

    def import_czi_fiducials(self, auto=False):
        """Scan a CZI's ZEN Shuttle & Find calibration markers (stage µm) and place
        them on the Fiducials layer. Called automatically on CZI load; appends to
        any current fiducials (4 px dedup) so it's idempotent across reloads.
        ``auto=True`` suppresses the not-a-CZI / read-error chatter."""
        if self.image_path is None or not czi_io.is_czi(self.image_path):
            if not auto:
                self.log_msg("⚠️ Import works on a CZI source (Shuttle & Find markers).")
            return
        try:
            markers = (czi_io.read_shuttle_and_find_markers(self.image_path)
                       .get("markers") or [])
        except Exception:
            if not auto:
                self.log_msg("❌ couldn't read CZI metadata:\n" + traceback.format_exc())
            return
        if not markers:
            self.log_msg("ℹ️ No ZEN Shuttle & Find fiducials found in this CZI.")
            return
        coords = ", ".join(f"({m['stage_x_um']:.1f}, {m['stage_y_um']:.1f})µm"
                           for m in markers)
        if self.geom is None or self.geom.stage_center_um is None:
            self.log_msg(f"⚠️ Found {len(markers)} Shuttle & Find fiducial(s) [{coords}] "
                         f"but this CZI has no stage anchor (scene CenterPosition / "
                         f"multi-scene) — can't place them on the image.")
            return
        # markers (stage µm) -> full-res px -> overview px (fid layer stores y,x)
        new_yx = []
        for m in markers:
            f = self.geom.stage_um_to_full(m["stage_x_um"], m["stage_y_um"])
            if f is None:
                continue
            ox, oy = self.geom.full_to_ds(float(f[0]), float(f[1]))
            new_yx.append((float(oy), float(ox)))
        if not new_yx:
            self.log_msg(f"⚠️ Found {len(markers)} Shuttle & Find fiducial(s) but "
                         f"couldn't map them to pixels."); return
        if self.fid_layer is None or self.fid_layer not in self.viewer.layers:
            self._ensure_edit_layers(self.current_polygons_xy())
        existing = (np.asarray(self.fid_layer.data, dtype=float).reshape(-1, 2)
                    if len(self.fid_layer.data) else np.empty((0, 2)))
        tol = 4.0  # overview px — re-import is idempotent
        added = []
        for (oy, ox) in new_yx:
            # compare against pre-existing fiducials AND ones added this batch
            pool = (np.vstack([existing, np.asarray(added, dtype=float)])
                    if added else existing)
            if pool.size and np.any(np.hypot(pool[:, 0] - oy, pool[:, 1] - ox) < tol):
                continue
            added.append((oy, ox))
        if not added:
            self.log_msg(f"ℹ️ Found {len(markers)} ZEN Shuttle & Find fiducial(s) "
                         f"[{coords}] — already present in the Fiducials layer.")
            return
        self.fid_layer.data = (np.vstack([existing, np.asarray(added, dtype=float)])
                               if existing.size else np.asarray(added, dtype=float))
        try:
            self.save_project()
        except Exception:
            pass
        xform = ("transposed axes" if self.geom.swap_xy else "direct axes")
        self.log_msg(f"✅ Found {len(markers)} ZEN Shuttle & Find fiducial(s) and "
                     f"imported {len(added)} into the Fiducials layer [{coords}] "
                     f"(stage→pixel via scene CenterPosition, {xform}). "
                     f"If they land on the wrong corners, tell me and I'll adjust the "
                     f"stage↔pixel transform.")


def main():
    # Avoid the in-repo sam2/ dir shadowing the installed `sam2` package: never
    # run from the repo root. All app paths are absolute, so this is safe.
    try:
        os.chdir(os.path.expanduser("~"))
    except Exception:
        pass
    viewer = napari.Viewer()
    gui = SectionIdentificationGUI(viewer)

    # New 4-tab workflow shell (Sections=this GUI, + ROIs/QC/Reorder + section
    # table). Guarded: if anything in the expansion fails to construct, fall back
    # to the original single-dock layout so the core detector always launches.
    attached = False
    try:
        from section_identification import stages
        stages.attach_workflow(viewer, gui)
        attached = True
    except Exception:
        traceback.print_exc()

    if not attached:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # Scroll vertically only — content reflows to the panel width instead of
        # forcing a horizontal scrollbar (keeps the right dock narrow & readable).
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(gui)
        scroll.setMinimumWidth(230)    # allow a narrow dock; long text wraps
        dock = viewer.window.add_dock_widget(scroll, name="STiM", area="right")
        # Open narrow, then release the cap so the user can still drag it.
        try:
            dock.setMaximumWidth(320)
            QTimer.singleShot(300, lambda: dock.setMaximumWidth(16777215))
        except Exception:
            pass
    napari.run()


if __name__ == "__main__":
    main()
