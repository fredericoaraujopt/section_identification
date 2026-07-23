"""Per-stage help — robust, technical, engineering-notebook-style reference.

Active voice, concrete mechanism, worked numbers; accessible without prior SAM/
SIFT knowledge (same tone as param_guide.py). Shown in a QDialog from each stage's
"❔" button.
"""

from __future__ import annotations

HELP = {
    "qc": """
<h2>Quality control</h2>
<p>Each detected section is scored on a downscaled, mask-confined crop (the
section interior only, so wafer background can't contaminate the measurement).
Every detector returns a continuous <b>severity</b> mapped to a 0–1 score via
<code>score = clip(severity / reference, 0, 1)</code>, and a boolean flag at the
threshold. The raw features are stored, so re-thresholding is instant — no
recompute.</p>
<h3>Detectors</h3>
<ul>
<li><b>Debris</b> — bright high-contrast specks: robust intensity outliers
(<code>median + k·MAD</code>) plus Laplacian-of-Gaussian blob count. Severity =
outlier-area fraction (reference ≈ 1% of the section).</li>
<li><b>Folds</b> — dark elongated ridges from doubled material: a Frangi
vesselness filter, kept only where components are elongated. Severity =
longest ridge / section minor-axis (a fold spanning the section ≈ 1.0).</li>
<li><b>Shredding</b> — tearing/fragmentation: connected-component count,
solidity (area / convex-hull area), holes (Euler number), and area vs the wafer
median. Severity = the worst of these.</li>
<li><b>Chattering</b> — periodic knife-vibration ripples: a Hann-windowed 2-D
FFT; severity = the prominence of an off-DC spectral peak above the band median
(low-sample radii are skipped so a clean section can't spike).</li>
</ul>
<p>Toggle <i>Show diagnostic overlay</i> and click a section in the table to see
the exact feature map that produced its dominant flag on the wafer.</p>
""",
    "rois": """
<h2>ROIs &amp; mFOVs</h2>
<p>This stage has three sub-tabs — <b>ROI</b> (place a region of interest on every
section), <b>Focus</b> (autofocus support points), and <b>mFOV</b> (tile grid
preview + CZI read). Mark fiducials first (Sections tab → CZI Shuttle &amp; Find,
or the 'm' key) — they anchor the pixel↔stage-µm transform used for every write.</p>

<h3>Placing ROIs — three methods</h3>
<p>Draw one polygon on the <i>ROI draft</i> layer (or SAM-assist trace it), then
pick how it reaches the other sections:</p>
<ol>
<li><b>Define + propagate</b> (pose) — the drafted ROI is stored in the reference
section's pose-normalised (upright, centred) frame and mapped onto every section
through its recovered pose, so it lands on the same anatomical region regardless
of rotation. <b>Fit</b> handles coming-in sections: <code>full</code> scales to the
section extent, <code>percent</code> to a fraction, <code>clip</code> intersects
with the section. Best when sections share a consistent geometry.</li>
<li><b>Propagate ROI to section centers</b> — drops a copy of the ROI, unchanged in
size and orientation, on each section's centroid. Use when sections vary widely in
size and pose-based fitting misplaces the ROI.</li>
<li><b>Automatic ROI detection (SAM)</b> — when the ROI is visually distinct from
the resin, SAM finds it inside each section (below).</li>
</ol>

<h3>Automatic detection — how SAM samples each section</h3>
<p>The drawn template tells SAM the ROI's expected <i>size and shape</i>; SAM finds
<i>where</i> it is in each section. Per section:</p>
<ol>
<li><b>Embed once.</b> The section's bounding box (plus <i>crop margin</i>) is read
as a crop, downscaled so its long side ≈ 1024 px, and encoded by SAM — one pass,
the expensive step.</li>
<li><b>Distribute a point grid.</b> A <code>points_per_side × points_per_side</code>
grid is laid across the section bbox (the same grid SAM's automatic detector uses)
and clipped to the section polygon, so SAM is prompted only <i>inside</i> the
section. This is exactly what the <b>Preview grid on sections</b> overlay shows —
double-click a table row to inspect one section's grid.</li>
<li><b>Predict at each point.</b> Each grid point is a cheap decoder prompt
returning up to 3 candidate masks (whole/part/subpart) with SAM's predicted IoU and
a stability score. Masks below <i>pred IoU</i> or <i>stability</i> are dropped.</li>
<li><b>Keep the one that matches the template.</b> Every surviving mask is scored on
SAM confidence + area closeness to the template (the <i>min/max area × template</i>
band) + shape overlap with the template + how well it sits inside the section. The
single best-scoring mask becomes that section's ROI; if none clears the
<i>score floor</i>, the section keeps its propagated-template ROI (fallback), so
every section always ends up with an ROI.</li>
</ol>
<p><b>Calibrate from template</b> populates every parameter from the template's
dimensions — grid density so several points land on the ROI, quality gates from the
ROI's apparent size, and the area band around the template area — the ROI analogue
of the section detector's calibration. Tune any value afterwards; the preview
updates live. <b>Contour</b> chooses whether the ROI outline is SAM's mask boundary
or the template shape re-fitted to the mask (uniform shape across sections).</p>

<h3>Writing to CZI</h3>
<p><i>File → Export</i>: sections become annotation <code>&lt;Layers&gt;</code>;
each ROI becomes a <code>&lt;TileRegion&gt;</code> (stage µm) with a tile grid
(Columns×Rows from your tile-µm) and a focus <code>&lt;SupportPoints&gt;</code>
grid — the exact nodes ZEN reads to place mFOVs and autofocus.</p>
""",
    "reorder": """
<h2>Reorder &amp; imaging order</h2>
<h3>Serial order (SIFT)</h3>
<p>Sections are scrambled on the wafer. Consecutive physical sections share fine
structure (vasculature, cell layout), so we match every pair with SIFT — which is
rotation- and scale-invariant, so arbitrary on-wafer rotation needs no
pre-alignment. The similarity between two sections is the number of
RANSAC-verified inlier matches. SIFT runs at <b>full resolution</b> (blood
vessels must be resolved); descriptors and the similarity matrix are cached so
re-running is instant.</p>
<p>The serial order is recovered by spectral seriation (the Fiedler ordering of
the similarity graph) refined by 2-opt, then oriented so the most isolated end is
first. Match lines (coloured by confidence) and the recovered chain are drawn on
the wafer; low-confidence joins are the ones to inspect.</p>
<h3>Imaging route (TSP)</h3>
<p>Given section centroids in stage µm, an open-path TSP (nearest-neighbour +
2-opt) minimises total stage travel. The route is drawn as directed vectors with
per-leg travel. On export, section/ROI IDs are renumbered to this order, so a ZEN
acquisition (which images by ID) follows the optimised path.</p>
""",
}


def show_help(parent, stage: str):
    """Open a non-modal help dialog for ``stage``. Returns the dialog (keep a
    reference)."""
    from qtpy.QtWidgets import QDialog, QTextBrowser, QVBoxLayout
    dlg = QDialog(parent)
    dlg.setWindowTitle(f"STiM — {stage} help")
    lay = QVBoxLayout(dlg)
    tb = QTextBrowser()
    tb.setOpenExternalLinks(True)
    tb.setHtml(HELP.get(stage, "<p>No help available.</p>"))
    lay.addWidget(tb)
    dlg.resize(560, 620)
    dlg.show()
    return dlg
