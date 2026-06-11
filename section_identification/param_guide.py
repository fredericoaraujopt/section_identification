"""In-app parameter guide for the automatic detector.

An objective, engineering-level reference for the SAM2 automatic-mask-generation
pipeline and every exposed parameter — definition, a worked example, and a
formula/units where it adds information. Grounded in the SAM2 source
(``sam2/automatic_mask_generator.py``, ``sam2/utils/amg.py``,
``sam2/utils/transforms.py``).

Each Advanced control has a ``?`` that opens this guide at that parameter
(``open_param_guide(parent, anchor="points_per_side")``).
"""

from __future__ import annotations

GUIDE_HTML = """
<style>
 body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 13px;
        margin: 10px 16px; }
 h2 { color:#7fd1ff; margin-top:20px; border-bottom:1px solid #444; padding-bottom:3px; }
 h3 { color:#ffd479; margin-top:16px; margin-bottom:2px; }
 code { background:#2a2a2a; padding:1px 4px; border-radius:3px; }
 .calc { background:#1e2a1e; border-left:3px solid #6cc06c; padding:6px 10px;
         margin:6px 0; font-family: ui-monospace, Menlo, monospace; font-size:12px; }
 .eg { color:#bbb; margin:4px 0; }
 .u { color:#9cf; }
 table { border-collapse:collapse; margin:6px 0; }
 td,th { border:1px solid #444; padding:3px 8px; text-align:left; }
</style>

<h2><a name="top"></a>How the automatic mask generator works</h2>
<p>SAM has three parts: an <b>image encoder</b> (a ViT/Hiera backbone) that turns
the image into an embedding once; a <b>prompt encoder</b>; and a lightweight
<b>mask decoder</b> that, given the embedding + a prompt (here, a point), outputs
candidate masks and a <i>predicted-IoU</i> quality score for each. The automatic
generator (AMG) drives this with no user clicks, as a fixed pipeline:</p>
<ol>
<li><b>Resize.</b> The image is resized so its long side = <b>1024 px</b>, then
encoded (<code>transforms.py</code>, <code>resolution=1024</code>).</li>
<li><b>Prompt grid.</b> A regular <code>points_per_side</code> × <code>points_per_side</code>
grid of points is placed on the image; each point is one prompt.</li>
<li><b>Decode.</b> For each point the decoder returns up to 3 masks (it predicts
several to resolve whole-object vs part ambiguity) + a predicted IoU each.
Points are run in batches of <code>points_per_batch</code>.</li>
<li><b>Quality filter.</b> Drop masks with predicted IoU &le; <code>pred_iou_thresh</code>,
then masks with stability &lt; <code>stability_score_thresh</code>.</li>
<li><b>De-duplicate (NMS).</b> Non-maximum suppression on the mask bounding boxes
(<code>box_nms_thresh</code>) keeps the best of each overlapping group.</li>
<li><b>Optional sub-cropping.</b> If <code>crop_n_layers</code> &gt; 0, repeat
steps 2–5 on overlapping sub-crops and merge (<code>crop_nms_thresh</code>).</li>
<li><b>Clean up.</b> Remove mask regions/holes smaller than
<code>min_mask_region_area</code>.</li>
</ol>
<p>Because step 1 always resizes to 1024, an object's size <i>in that 1024 frame</i>
determines whether the decoder can resolve it:</p>
<div class="calc">section_to_SAM = section_px &times; 1024 / processed_long_side_px
&nbsp;&nbsp;<span class="u"># px</span><br>
usable &ge; ~16 px · unreliable &lt; 8 px</div>
<p class="eg"><b>Example.</b> 200-px sections read in a 4096-px overview →
200·1024/4096 = 50 px to SAM (good). The same sections in the native 20000-px
image → 10 px (poor): tile the image so each tile, upscaled to 1024, enlarges the
section.</p>

<h2><a name="differences"></a>Which control is which</h2>
<p><b>tile px vs crop layers.</b> <code>tile_px</code> is STiM's outer split: the
image is processed one tile at a time and each tile is resized to 1024.
<code>crop_n_layers</code> is SAM's inner re-cropping <i>within</i> one tile. Both
enlarge small objects; set <code>tile_px</code> first.</p>
<p><b>tile px vs overview px.</b> <code>overview_long_side</code> is the resolution
the image is read at — the only source of real detail. <code>tile_px</code> only
re-partitions pixels already read; it cannot add detail beyond the overview.</p>
<p><b>min mask area vs min section area.</b> <code>min_mask_region_area</code>
removes specks/holes <i>inside</i> a single mask (SAM step 7).
<code>min section area</code> discards whole <i>detections</i> below a size and
feeds the area-DBSCAN (STiM post-filter).</p>
<p><b>target → SAM</b> is not applied at run time; Calibrate uses it to compute
<code>tile_px</code> and <code>points_per_side</code>.</p>

<h2><a name="choosing-from-size"></a>Setting parameters from section size</h2>
<p>With section diameter <code>d</code> (px) and tile long side <code>L</code> (px):</p>
<div class="calc">
points_per_side = clip( ceil( 2.5 · L / d ), 16, 128 )<br>
tile_px         = round( 1024 · d / target_to_SAM )<br>
min_mask_region_area = round( 0.05 · section_area_px )<br>
pred_iou / stability = 0.85/0.96 (section_to_SAM&ge;20) | 0.80/0.95 (&ge;10) | 0.75/0.92 (else)
</div>

<h2><a name="memory"></a>Memory usage</h2>
<p>Peak memory is dominated by decoding one batch: <code>points_per_batch</code>
points × 3 candidate masks, each at the tile resolution, in float32, with ~3.5×
for intermediates.</p>
<div class="calc">
peak_bytes ≈ points_per_batch · 3 · tile_h · tile_w · 4 · 3.5<br>
safe points_per_batch = budget_bytes / (tile_h · tile_w · 4 · 3.5 · 3)
</div>
<p class="eg"><b>Example.</b> 1024×1024 tile, 1 GB budget →
1e9 / (1024·1024·4·3.5·3) ≈ <b>22</b> points per batch. A 3072×3072 tile needs
~9× less (≈2–3). <code>points_per_batch</code> changes only memory and speed, not
the result; STiM auto-caps it to the host.</p>

<h2><a name="runtime"></a>Runtime</h2>
<p>runtime ≈ tiles × seconds_per_pass. Approximate seconds per 1024-px pass:</p>
<table><tr><th></th><th>CPU</th><th>Apple&nbsp;MPS</th><th>CUDA</th></tr>
<tr><td>hiera_tiny</td><td>~12 s</td><td>~1.5 s</td><td>~0.3 s</td></tr>
<tr><td>hiera_base_plus</td><td>~45 s</td><td>~4 s</td><td>~0.9 s</td></tr></table>
<p><code>crop_n_layers=1</code> multiplies passes by ~5; <code>use_m2m</code> by ~2.
Runtime scales with <code>points_per_side²</code>.</p>

<h2>Parameters</h2>

<h3><a name="points_per_side"></a>points_per_side <span class="u">(count, per tile)</span></h3>
<p>Sets the side length of the square grid of point prompts placed on each tile
(total = points_per_side²). Grid spacing is uniform.</p>
<p class="eg"><b>Example.</b> points_per_side=32 on a tile places 1024 prompts;
spacing in the 1024 frame = 1024/32 = 32 px, so a 40-px (to-SAM) section gets ~1
point. Raising to 48 gives ~1.5 → more reliable hits on smaller sections.</p>
<div class="calc">grid_spacing_1024 = 1024 / points_per_side<br>
to land ≥ k points on a section: points_per_side ≥ k · tile_px / section_px</div>

<h3><a name="pred_iou_thresh"></a>pred_iou_thresh <span class="u">(predicted IoU, 0–1; default 0.80)</span></h3>
<p>The decoder predicts, for each mask, the IoU it expects against the true object.
The AMG keeps a mask only if that predicted score exceeds this threshold. It is the
model's self-rated quality, not a measured overlap.</p>
<p class="eg"><b>Example.</b> 0.80 discards any mask the model rates below 0.80
quality. Lowering to 0.75 recovers faint/low-contrast sections at the cost of more
false positives; raising to 0.88 keeps only high-confidence masks.</p>

<h3><a name="stability_score_thresh"></a>stability_score_thresh <span class="u">(ratio, 0–1; default 0.95)</span></h3>
<p>Measures how much a mask changes when the binarisation cutoff is perturbed by
±<code>stability_score_offset</code>. A mask with a crisp boundary barely changes
(score near 1); a fuzzy one shrinks/grows a lot. Masks below the threshold are
dropped.</p>
<div class="calc">stability = |logits &gt; T+offset| / |logits &gt; T−offset|
&nbsp;<span class="u"># pixel-count ratio, T = mask_threshold</span></div>
<p class="eg"><b>Example.</b> Lower to 0.90–0.92 for small or low-contrast sections
(naturally fuzzier edges); 0.96 for crisp, high-contrast sections.</p>

<h3><a name="stability_score_offset"></a>stability_score_offset <span class="u">(logits; default 1.0)</span></h3>
<p>The perturbation applied to the mask cutoff when computing stability (the mask
decoder outputs logits; <code>mask_threshold</code> = 0). A larger offset is a
stricter stability test. Rarely changed.</p>

<h3><a name="box_nms_thresh"></a>box_nms_thresh <span class="u">(box IoU, 0–1; default 0.70)</span></h3>
<p>Controls de-duplication = non-maximum suppression (NMS): of any group of masks
whose bounding boxes overlap by more than this IoU, only the one with the highest
predicted IoU is kept; the rest are discarded. (The grid + 3-masks-per-point
produce many near-identical masks; NMS removes the redundancy.)</p>
<p class="eg"><b>Example.</b> 0.70 drops a mask whose box overlaps a kept mask's box
by &gt;70% (IoU). Lower (0.5) removes more aggressively; raise (0.85) to keep two
adjacent/touching sections that share part of a bounding box.</p>

<h3><a name="crop_n_layers"></a>crop_n_layers <span class="u">(count; default 0)</span></h3>
<p>Re-runs the whole grid-and-decode pipeline on overlapping sub-crops of the tile,
then merges. Layer <i>i</i> adds (2^i)² crops (layer 1 → 4 crops in a 2×2 grid;
layer 2 → 16). Each crop is itself resized to 1024, so objects inside it appear
larger and small-object recall improves — at roughly 5× the passes for layer 1.</p>
<p class="eg"><b>Example.</b> On a 1024-px tile, crop_n_layers=1 also processes four
~512-px crops upscaled to 1024, doubling the apparent size of objects in them. Only
needed when tiling alone can't bring a section to ≥16 px to SAM.</p>

<h3><a name="crop_overlap_ratio"></a>crop_overlap_ratio <span class="u">(fraction; default 0.34)</span></h3>
<p>Sets how far neighbouring sub-crops overlap, so a section on a crop seam still
lies wholly inside one crop.</p>
<div class="calc">overlap_px = round( crop_overlap_ratio · short_side · 2/n_per_side )
&nbsp;<span class="u"># layer 1 (n=2): overlap_px ≈ ratio · short_side</span></div>
<p class="eg"><b>Example.</b> A 1024-px crop short side at ratio 0.34 overlaps by
~348 px — enough for any section &lt;348 px. Increase if large sections fall on
seams.</p>

<h3><a name="crop_n_points_downscale_factor"></a>crop_n_points_downscale_factor <span class="u">(integer; default 1)</span></h3>
<p>Divides <code>points_per_side</code> on each deeper crop layer (layer <i>i</i>
uses points_per_side / factor^i), since sub-crops cover less area. 2 is typical
when <code>crop_n_layers</code> ≥ 1.</p>

<h3><a name="min_mask_region_area"></a>min_mask_region_area <span class="u">(px², processed frame; default 0)</span></h3>
<p>After NMS, removes disconnected mask regions and fills holes smaller than this
many pixels (via connected components). A specks/pinholes cleaner operating inside
each mask, in the tile's pixel units.</p>
<div class="calc">min_mask_region_area ≈ 0.05 · section_area_px</div>
<p class="eg"><b>Example.</b> 200-px sections (area ≈ 40000 px²) → ~2000 px² removes
specks under 5% of a section while keeping the sections.</p>

<h3><a name="min_section_area"></a>min section area <span class="u">(px², processed frame)</span></h3>
<p>STiM discards any whole detection smaller than this, and uses it to anchor the
area-DBSCAN band. Calibrate sets it to ~half the median section area.</p>

<h3><a name="use_m2m"></a>use_m2m <span class="u">(on/off; default off)</span></h3>
<p>Adds a second decoder pass that feeds each first-pass mask back as a prompt to
refine it. Improves boundary quality; roughly doubles decoder calls.</p>

<h3><a name="tile_px"></a>tile_px <span class="u">(px, image frame; 0 = whole image)</span></h3>
<p>STiM splits the read image into roughly equal tiles of this size (with overlap),
segments each independently, and streams results. SAM resizes each tile to 1024,
so the section's apparent size to SAM scales inversely with tile_px.</p>
<div class="calc">section_to_SAM = section_px · 1024 / tile_px</div>
<p class="eg"><b>Example.</b> 120-px sections, tile_px=1920 → 64 px to SAM. Halving
tile_px to 960 → 128 px to SAM (sharper) but ~4× the tiles.</p>

<h3><a name="overlap"></a>tile overlap <span class="u">(fraction, 0–1)</span></h3>
<p>Fractional overlap between STiM tiles (not SAM crops). A section touching an
internal tile edge is dropped to avoid duplicates, so the overlap must exceed a
section so it sits whole inside a neighbouring tile.</p>

<h3><a name="target_sam_px"></a>target → SAM <span class="u">(px)</span></h3>
<p>The intended apparent section size to SAM. Calibrate solves
<code>tile_px = round(1024 · section_px / target)</code> and the grid from it. Not
applied directly at run time. Higher target → smaller tiles → sharper masks and
more/slower tiles. ~64 is a good balance.</p>

<h3><a name="overview_long_side"></a>overview long side <span class="u">(px; reload to apply)</span></h3>
<p>The long-side resolution the source image is read at (the CZI pyramid level, or
downscale for other formats). This is the only source of real spatial detail —
tiling cannot exceed it. Larger = more detail at higher memory/time; the host caps
it.</p>

<h3><a name="points_per_batch"></a>points_per_batch <span class="u">(count; memory/speed only)</span></h3>
<p>Number of grid points decoded together in one forward pass. Affects peak memory
and throughput only — never the result. See <a href="#memory">Memory usage</a> for
the formula. Lower to avoid out-of-memory or swapping; raise on a large GPU.</p>

<h3><a name="model"></a>model <span class="u">(hiera tiny | small | base_plus | large)</span></h3>
<p>The SAM2 Hiera backbone size. Larger backbones give higher segmentation accuracy
at more VRAM and time. STiM auto-selects by host: tiny/small on CPU or low memory,
base_plus/large on a capable GPU.</p>

<h3><a name="device"></a>device <span class="u">(cpu | cuda | mps)</span></h3>
<p>Where inference runs. Auto resolves CUDA &gt; Apple MPS &gt; CPU. CUDA is fastest;
CPU works anywhere but is much slower (pair with a tiny model and larger sections).</p>

<h3><a name="dbscan"></a>area DBSCAN <span class="u">(STiM post-filter)</span></h3>
<p>After detection, clusters all detections by their area (1-D DBSCAN over area) and
keeps the largest cluster, discarding size outliers (debris, merged clumps). Useful
when sections are similarly sized.</p>
"""


def open_param_guide(parent=None, anchor=None):
    """Open (or raise) the parameter-guide dialog, scrolled to ``anchor``."""
    from qtpy.QtWidgets import QDialog, QVBoxLayout, QTextBrowser

    dlg = getattr(open_param_guide, "_dialog", None)
    if dlg is None:
        dlg = QDialog(parent)
        dlg.setWindowTitle("STiM — Parameter guide")
        dlg.resize(600, 760)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setOpenLinks(False)
        browser.document().setDocumentMargin(14)   # keep text off the window edge
        browser.setHtml(GUIDE_HTML)
        browser.anchorClicked.connect(
            lambda url: browser.scrollToAnchor(url.toString().lstrip("#")))
        lay.addWidget(browser)
        open_param_guide._dialog = dlg
        open_param_guide._browser = browser

    browser = open_param_guide._browser
    dlg.show(); dlg.raise_(); dlg.activateWindow()
    browser.scrollToAnchor(anchor or "top")
    return dlg
