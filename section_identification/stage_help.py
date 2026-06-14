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
<p>Define one region of interest on a reference section; it is stored in that
section's <b>pose-normalised</b> (upright, centred) frame, then propagated to
every section through each section's recovered pose — so it lands on the same
anatomical region regardless of how the section is rotated on the wafer.</p>
<h3>Workflow</h3>
<ol>
<li>Draw one polygon on the <i>ROI draft</i> layer (inside a section).</li>
<li><i>Define + propagate</i>: the section under the ROI becomes the reference;
the ROI maps onto all sections.</li>
<li><b>Fit</b> coming-in (smaller/partial) sections: <code>full</code> scales the
ROI to the section extent; <code>percent</code> to a fraction; <code>clip</code>
intersects it with the section so mFOVs never image empty resin.</li>
<li><i>Write into CZI</i>: sections become annotation <code>&lt;Layers&gt;</code>;
each ROI becomes a <code>&lt;TileRegion&gt;</code> (stage µm) with a tile grid
(Columns×Rows from your tile-µm) and a focus <code>&lt;SupportPoints&gt;</code>
grid — the exact nodes ZEN reads to place mFOVs and autofocus.</li>
</ol>
<p>Mark fiducials first (Sections tab → CZI Shuttle &amp; Find, or the 'm' key) —
they anchor the pixel↔stage-µm transform used for every stage-µm write.</p>
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
