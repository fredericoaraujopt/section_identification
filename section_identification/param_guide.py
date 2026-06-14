"""In-app parameter guide for the automatic detector.

An objective, engineering-level reference for the SAM2 automatic-mask-generation
pipeline and every exposed parameter — definition, a worked example, and a
formula/units where it adds information. Grounded in the SAM2 source
(``sam2/automatic_mask_generator.py``, ``sam2/utils/amg.py``,
``sam2/utils/transforms.py``).

The parameters are organised into the SAME six effect-clusters as the Advanced
panel in the GUI: each cluster has one primary lever (★) and the rest are coupled
to it. Each Advanced control has a ``?`` that opens this guide at that parameter
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
 .lever { color:#9cffb0; font-weight:bold; }
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
<li><b>Resize.</b> SAM resizes the image so its long side = <b>1024 px</b>, then
encodes it (<code>transforms.py</code>, <code>resolution=1024</code>).</li>
<li><b>Prompt grid.</b> SAM places a regular <code>points_per_side</code> × <code>points_per_side</code>
grid of points over the image; each point is one prompt.</li>
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
<p>Step 1 always resizes the processed image (one tile) to 1024&nbsp;px, so what
counts is how many of those 1024 pixels a section spans — not how many pixels it had
in the source. That apparent size decides whether the decoder can resolve the section
at all:</p>
<div class="calc">section_to_SAM = section_px &times; 1024 / processed_long_side_px
&nbsp;&nbsp;<span class="u"># px in SAM's 1024 frame</span><br>
reliable &ge; ~16&nbsp;px&nbsp;&middot;&nbsp;unreliable &lt; 8&nbsp;px</div>
<p>Read that formula carefully — it has a consequence that surprises most people.
When you process the <b>whole</b> wafer as a single tile, <code>section_px</code> and
<code>processed_long_side_px</code> grow together as you raise the overview
resolution, so their ratio — and therefore <code>section_to_SAM</code> — does
<i>not</i> change. Reading a finer overview, on its own, buys no apparent size.
<b>Tiling</b> is what buys it: each tile covers a smaller slice of the wafer yet still
expands to 1024, so the section fills more of SAM's frame.</p>
<p class="eg"><b>Example.</b> Take a section that is 2% of the wafer's width. Processed
as one whole-image tile it reaches SAM at 0.02&nbsp;&times;&nbsp;1024&nbsp;&asymp;&nbsp;20&nbsp;px
<i>whether you read the overview at 2048&nbsp;px or 8192&nbsp;px</i> — identical masks,
but the 8192 read costs far more (see Runtime). Now split the same wafer into a
4&nbsp;&times;&nbsp;4 grid: each section spans 8% of its tile, so it reaches SAM at
0.08&nbsp;&times;&nbsp;1024&nbsp;&asymp;&nbsp;82&nbsp;px — four times sharper. Overview
px sets the <i>ceiling</i> on real detail; tiling is what spends it.</p>

<h2><a name="clusters"></a><a name="differences"></a>Parameter clusters (which knob is the lever)</h2>
<p>The ~18 knobs collapse into six groups. Within a group the knobs mostly move the
<i>same</i> outcome, so tune the <span class="lever">★ lever</span> first and leave the
rest unless you have a reason — tuning two coupled knobs at once compounds.</p>
<table>
<tr><th>Cluster</th><th>★ Lever</th><th>Also in it (coupled)</th></tr>
<tr><td><b>1 · Resolution</b> — how big SAM sees a section</td>
    <td>overview px</td><td>tile px, crop layers</td></tr>
<tr><td><b>2 · Coverage</b> — query-point density</td>
    <td>points / side</td><td>crop grid ÷</td></tr>
<tr><td><b>3 · Quality gates</b></td>
    <td>pred IoU</td><td>stability, stability offset, refine (use_m2m)</td></tr>
<tr><td><b>4 · Deduplication</b></td>
    <td>box NMS</td><td>tile overlap, crop overlap</td></tr>
<tr><td><b>5 · Keep / drop by size</b></td>
    <td>area DBSCAN</td><td>min section area, min mask area</td></tr>
<tr><td><b>6 · Performance</b> — no effect on masks</td>
    <td>model</td><td>points / batch, low-memory, device</td></tr>
</table>
<p>The coupling that bites most often:</p>
<ul>
<li><b>Resolution = supply &times; spend.</b> <code>overview_long_side</code> is the only
knob that adds real detail — it reads a finer source/CZI level and sets the ceiling on
what any later step can recover. But SAM only <i>uses</i> that detail when you
<code>tile</code> or <code>crop</code>: each tile expands to 1024, so a smaller tile
enlarges the section in SAM's frame. Raising overview px while still processing the
whole image as one tile adds cost without adding apparent size. Pair a finer overview
with finer tiling to turn supply into resolution — and don't tile finer than the
overview supports, or you only upscale blur.</li>
<li><b>Tiles are 1024-native; the overview follows.</b> SAM's tile should be 1024&nbsp;px
so it maps 1:1 into the encoder with no upscaling. Calibration therefore reads the
overview at <b>N·1024</b>: <code>N=1</code> (overview 1024) processes the whole image in
one tile; <code>N&gt;1</code> reads a finer level and cuts <code>N×N</code> tiles of 1024,
which is the <i>only</i> way to add real apparent size to a section too small to resolve
whole. Raising the overview without tiling buys nothing; tiling below 1024 only upscales
blur. See <a href="#choosing-from-size">Setting parameters from section size</a>.</li>
<li><b>Calibration sets tile px + points/side:</b> from the example section it sizes
<code>tile_px</code> from the resolve floor (tile only when a section can't be resolved
whole) and <code>points_per_side</code> from a &ge;2-points-per-section target. Set
<code>tile_px</code> / <code>points_per_side</code> directly to override.</li>
<li><b>Two quality gates overlap.</b> <code>pred_iou_thresh</code> and
<code>stability_score_thresh</code> measure different things but a bad mask usually
trips both — raising either tightens output similarly. <code>stability_score_offset</code>
is a sub-knob of stability, not independent.</li>
<li><b>Three size filters.</b> <code>min_mask_region_area</code> (inside a mask, tile px),
<code>min section area</code> (whole detections, overview px) and <code>area DBSCAN</code>
all drop small things at different stages — DBSCAN can subsume the manual floor.</li>
<li><b>points_per_batch changes nothing in the result</b> — it is purely a memory/speed
dial (see Memory usage).</li>
</ul>

<h2><a name="choosing-from-size"></a>Setting parameters from section size</h2>
<p>SAM's tile is pinned to <b>1024&nbsp;px</b> (1:1 into the encoder, no upscaling), so the
two knobs calibration tunes are the <b>overview resolution</b> and
<code>points_per_side</code> — driven by the section's <b>thin (minor) axis</b>
<code>d</code> for resolvability and its <b>area</b> <code>A</code> for point count.</p>
<p><b>1 · Read the overview at N·1024.</b> The section's thin axis on a whole 1024-px image
is <code>a = d · 1024 / overview_px</code> (a property of the section, not the current
overview). If that clears SAM's resolvable floor (<code>resolve_px</code>&nbsp;&asymp;&nbsp;24&nbsp;px)
one 1024 tile suffices. Otherwise read the overview <code>N×</code> finer and cut
<code>N×N</code> tiles of 1024 — each tile magnifies the section by N:</p>
<div class="calc">
N = ceil( resolve_px / a )&nbsp;&nbsp;<span class="u"># a = d·1024/overview_px (thin axis, whole-image 1024)</span><br>
overview = N · 1024&nbsp;&middot;&nbsp;tile_px = 1024&nbsp;&middot;&nbsp;N×N tiles&nbsp;&nbsp;<span class="u"># N=1 ⇒ whole image</span>
</div>
<p>N is clamped to the source's true resolution (you can't read finer than the CZI has) and
to the host memory budget; for a flat source like a PNG there is no finer level, so the
overview is left as-is and a sub-1024 tile upscales as a last resort.</p>
<p><b>2 · Grid for &ge; 2 points per section.</b> The number of seed points that land on a
section follows a simple area law (measured 1.47 vs 1.45 predicted at a 32&times;32 grid).
Solve it for the target count <code>N_pts</code> (default 2), in the 1024 tile:</p>
<div class="calc">
points_on_section &asymp; A · (points_per_side / tile_px)²<br>
points_per_side = clip( ceil( tile_px · &radic;(N_pts / A) ), 16, 128 )&nbsp;&nbsp;<span class="u"># tile_px = 1024, N_pts = 2</span>
</div>
<p class="eg"><b>Example.</b> A section whose thin axis is ~30&nbsp;px on a whole 1024-px image
already clears 24, so <code>N=1</code>: <b>overview 1024, one tile</b>, and (area &asymp; 1500&nbsp;px²)
<code>points_per_side = ceil(1024·&radic;(2/1500)) &asymp; 38</code> → ~2 points per section. A
section only ~8&nbsp;px on the whole image needs <code>N = ceil(24/8) = 3</code>: <b>overview
3072, 3×3 tiles of 1024</b>, each enlarging the section to ~24&nbsp;px, with the grid sized
the same way per tile.</p>
<p>Because the total decode work is <code>overview² · N_pts / A</code> — the same whether you
use one dense-grid tile or many sparse ones — tiling never helps the point target; it costs
extra encoder passes. So STiM uses the <i>fewest</i> tiles that resolve the section (lowest
N) and lets the grid hit the 2-point target. The size band
(<code>min section area</code>, DBSCAN), <code>min_mask_region_area = 0.05·A</code> and the
size-keyed quality gates (<code>0.85/0.96</code> at &ge;20&nbsp;px, etc.) still apply.</p>

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
~9× less (≈2–3). When you process the whole image as one tile,
<code>tile_h · tile_w</code> <i>is</i> the overview px², so a finer overview both
raises peak memory and forces a smaller safe batch (which then slows the run).
<code>points_per_batch</code> changes only memory and speed, not the result; STiM
auto-caps it to the host.</p>

<h2><a name="runtime"></a>Runtime</h2>
<p>A run adds up three costs that scale differently — which is why it can slow down
sharply even when the tile count and the point grid stay fixed:</p>
<table>
<tr><th>Stage</th><th>Cost grows with</th></tr>
<tr><td>Encode each tile</td><td><b>constant</b> per tile — SAM always works at 1024&nbsp;px,
so the backbone ignores how many pixels the tile actually has</td></tr>
<tr><td>Decode the point grid</td><td><code>points_per_side²</code> &times; tiles —
one decoder pass per query point</td></tr>
<tr><td>Build &amp; clean masks<br>(upscale, stability, NMS, RLE)</td>
<td><b>tile area</b> &times; mask count — SAM upsamples every candidate mask to the
tile's resolution, so this term grows with <code>tile_px²</code></td></tr>
</table>
<p>For tiles near 1024&nbsp;px the encode and decode dominate and a simple estimate
holds — seconds per 1024-px tile at a 32&times;32 grid (scale by
<code>(points_per_side/32)²</code> for other grids):</p>
<table><tr><th></th><th>CPU</th><th>Apple&nbsp;MPS</th><th>CUDA</th></tr>
<tr><td>hiera_tiny</td><td>~12 s</td><td>~1.5 s</td><td>~0.3 s</td></tr>
<tr><td>hiera_base_plus</td><td>~45 s</td><td>~4 s</td><td>~0.9 s</td></tr></table>
<p>Above ~1024&nbsp;px per tile the mask-building term takes over and runtime climbs
with the tile's <i>area</i>. Processing a whole wafer as one 4096-px tile costs
roughly 16&times; a 1024-px tile <i>for the same sections and the same point grid</i>,
and it also forces <code>points_per_batch</code> down (see Memory usage), adding even
more passes. So when a run is slow, shrink the tile rather than enlarge the overview.
<code>crop_n_layers=1</code> multiplies the decode-and-mask work by ~5;
<code>use_m2m</code> by ~2. The live <b>Effect</b> readout shows the resulting tile
count and a rough time for your current settings.</p>

<!-- ================================================================= -->
<h2><a name="cluster-resolution"></a>1 · Resolution &nbsp;<span class="u">(how big SAM sees a section)</span></h2>
<p><span class="lever">★ lever: overview px.</span> The others only magnify what the
overview already captured.</p>

<h3><a name="overview_long_side"></a>★ overview long side <span class="u">(px; reload to apply)</span></h3>
<p>The long-side resolution the source image is read at (the CZI pyramid level, or
downscale for other formats). This is the only source of real spatial detail —
tiling cannot exceed it. Calibration sets it to <b>N·1024</b> so SAM's tiles are
1024&nbsp;px: 1024 = whole image, N·1024 = N×N tiles for sections too small to resolve
whole. Larger = more detail at higher memory/time; the host caps it.</p>

<h3><a name="tile_px"></a>tile_px <span class="u">(px, image frame; 0 = whole image; auto)</span></h3>
<p>STiM splits the read image into roughly equal tiles of this size (with overlap),
segments each independently, and streams results. SAM resizes each tile to 1024,
so the section's apparent size to SAM scales inversely with tile_px. It re-partitions
pixels already read — it cannot add detail beyond the overview. Calibration pins this to
<b>1024</b> (1:1 into the encoder) and instead raises the overview to N·1024 when a section
is too small to resolve whole, so each of the N×N tiles is a native 1024-px crop.</p>
<div class="calc">section_to_SAM = section_px · 1024 / tile_px&nbsp;&nbsp;<span class="u"># = section_px when tile_px = 1024</span><br>
N = ceil( resolve_px / (minor_px · 1024 / overview_px) )&nbsp;&nbsp;<span class="u"># overview = N·1024, N=1 ⇒ whole image</span></div>
<p class="eg"><b>Example.</b> 120-px sections, tile_px=1920 → 64 px to SAM. Halving
tile_px to 960 → 128 px to SAM (sharper) but ~4× the tiles. A 30-px-thin section in a
1024-px overview is already ~30&nbsp;px &ge; 24 to SAM, so it stays whole-image.</p>

<h3><a name="crop_n_layers"></a>crop_n_layers <span class="u">(count; default 0; magnify only)</span></h3>
<p>Re-runs the whole grid-and-decode pipeline on overlapping sub-crops of the tile,
then merges. Layer <i>i</i> adds (2^i)² crops (layer 1 → 4 crops in a 2×2 grid;
layer 2 → 16). Each crop is itself resized to 1024, so objects inside it appear
larger and small-object recall improves — at roughly 5× the passes for layer 1. Like
tile_px it only magnifies overview pixels (no new detail).</p>
<p class="eg"><b>Example.</b> On a 1024-px tile, crop_n_layers=1 also processes four
~512-px crops upscaled to 1024, doubling the apparent size of objects in them. Only
needed when tiling alone can't bring a section to ≥16 px to SAM.</p>

<!-- ================================================================= -->
<h2><a name="cluster-coverage"></a>2 · Coverage &nbsp;<span class="u">(query-point density)</span></h2>
<p><span class="lever">★ lever: points / side.</span> (tile px and crop layers also
change how many points land on a section.)</p>

<h3><a name="points_per_side"></a>★ points_per_side <span class="u">(count, per tile)</span></h3>
<p>Sets the side length of the square grid of point prompts placed on each tile
(total = points_per_side²). Grid spacing is uniform.</p>
<p class="eg"><b>Example.</b> points_per_side=32 on a tile places 1024 prompts;
spacing in the 1024 frame = 1024/32 = 32 px, so a 40-px (to-SAM) section gets ~1
point. Raising to 38 gives ~2 → the count STiM targets (one seed is enough to segment a
section, but two is robust to it falling between grid nodes).</p>
<div class="calc">points_on_section &asymp; section_area · (points_per_side / tile_px)²<br>
for &ge; N points: points_per_side &ge; tile_px · &radic;(N / section_area)&nbsp;&nbsp;<span class="u"># STiM uses N = 2</span></div>

<h3><a name="crop_n_points_downscale_factor"></a>crop_n_points_downscale_factor <span class="u">(integer; default 1)</span></h3>
<p>Divides <code>points_per_side</code> on each deeper crop layer (layer <i>i</i>
uses points_per_side / factor^i), since sub-crops cover less area. 2 is typical
when <code>crop_n_layers</code> ≥ 1; inert when crop layers = 0.</p>

<!-- ================================================================= -->
<h2><a name="cluster-quality"></a>3 · Quality gates</h2>
<p><span class="lever">★ lever: pred IoU.</span> stability is a second, overlapping gate.</p>

<h3><a name="pred_iou_thresh"></a>★ pred_iou_thresh <span class="u">(predicted IoU, 0–1; default 0.80)</span></h3>
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
dropped. Overlaps with pred IoU — a poor mask usually fails both.</p>
<div class="calc">stability = |logits &gt; T+offset| / |logits &gt; T−offset|
&nbsp;<span class="u"># pixel-count ratio, T = mask_threshold</span></div>
<p class="eg"><b>Example.</b> Lower to 0.90–0.92 for small or low-contrast sections
(naturally fuzzier edges); 0.96 for crisp, high-contrast sections.</p>

<h3><a name="stability_score_offset"></a>stability_score_offset <span class="u">(logits; default 1.0; sub-knob)</span></h3>
<p>The perturbation applied to the mask cutoff when computing stability (the mask
decoder outputs logits; <code>mask_threshold</code> = 0). A larger offset is a
stricter stability test. A refinement of the stability gate, not independent. Rarely changed.</p>

<h3><a name="use_m2m"></a>use_m2m <span class="u">(on/off; default off)</span></h3>
<p>Adds a second decoder pass that feeds each first-pass mask back as a prompt to
refine it. Improves boundary quality; roughly doubles decoder calls.</p>

<!-- ================================================================= -->
<h2><a name="cluster-dedup"></a>4 · Deduplication</h2>
<p><span class="lever">★ lever: box NMS.</span> tile/crop overlap prevent a section
straddling a seam from being split or duplicated.</p>

<h3><a name="box_nms_thresh"></a>★ box_nms_thresh <span class="u">(box IoU, 0–1; default 0.70)</span></h3>
<p>Controls de-duplication = non-maximum suppression (NMS): of any group of masks
whose bounding boxes overlap by more than this IoU, only the one with the highest
predicted IoU is kept; the rest are discarded. (The grid + 3-masks-per-point
produce many near-identical masks; NMS removes the redundancy.)</p>
<p class="eg"><b>Example.</b> 0.70 drops a mask whose box overlaps a kept mask's box
by &gt;70% (IoU). Lower (0.5) removes more aggressively; raise (0.85) to keep two
adjacent/touching sections that share part of a bounding box.</p>

<h3><a name="overlap"></a>tile overlap <span class="u">(fraction, 0–1)</span></h3>
<p>Fractional overlap between STiM tiles (not SAM crops). A section touching an
internal tile edge is dropped to avoid duplicates, so the overlap must exceed a
section so it sits whole inside a neighbouring tile.</p>

<h3><a name="crop_overlap_ratio"></a>crop_overlap_ratio <span class="u">(fraction; default 0.34; inert at 0 layers)</span></h3>
<p>Sets how far neighbouring sub-crops overlap, so a section on a crop seam still
lies wholly inside one crop.</p>
<div class="calc">overlap_px = round( crop_overlap_ratio · short_side · 2/n_per_side )
&nbsp;<span class="u"># layer 1 (n=2): overlap_px ≈ ratio · short_side</span></div>
<p class="eg"><b>Example.</b> A 1024-px crop short side at ratio 0.34 overlaps by
~348 px — enough for any section &lt;348 px. Increase if large sections fall on
seams.</p>

<!-- ================================================================= -->
<h2><a name="cluster-size"></a>5 · Keep / drop by size &nbsp;<span class="u">(3 overlapping filters)</span></h2>
<p><span class="lever">★ lever: area DBSCAN.</span> The two area floors below overlap with it.</p>

<h3><a name="dbscan"></a>★ area DBSCAN <span class="u">(STiM post-filter)</span></h3>
<p>After detection, clusters all detections by their area (1-D DBSCAN over area) and
keeps the largest cluster, discarding size outliers (debris, merged clumps). Useful
when sections are similarly sized.</p>

<h3><a name="min_section_area"></a>min section area <span class="u">(px², overview frame)</span></h3>
<p>STiM discards any whole detection smaller than this, and uses it to anchor the
area-DBSCAN band. Calibrate sets it to ~half the median section area.</p>

<h3><a name="min_mask_region_area"></a>min_mask_region_area <span class="u">(px², processed/tile frame; default 0)</span></h3>
<p>After NMS, removes disconnected mask regions and fills holes smaller than this
many pixels (via connected components). A specks/pinholes cleaner operating inside
each mask, in the tile's pixel units (so it scales with resolution — distinct from
min section area, which is in overview px).</p>
<div class="calc">min_mask_region_area ≈ 0.05 · section_area_px</div>
<p class="eg"><b>Example.</b> 200-px sections (area ≈ 40000 px²) → ~2000 px² removes
specks under 5% of a section while keeping the sections.</p>

<!-- ================================================================= -->
<h2><a name="cluster-performance"></a>6 · Performance &nbsp;<span class="u">(no effect on the masks)</span></h2>
<p><span class="lever">★ lever: model.</span> These set speed/memory only; the segmentation
is unchanged (except that low-memory drops to 1 mask/point).</p>

<h3><a name="model"></a>★ model <span class="u">(hiera tiny | small | base_plus | large)</span></h3>
<p>The SAM2 Hiera backbone size. Larger backbones give higher segmentation accuracy
at more VRAM and time. STiM auto-selects by host: tiny/small on CPU or low memory,
base_plus/large on a capable GPU.</p>
<p><b>Each variant is a SEPARATE checkpoint file</b> (<code>sam2.1_hiera_tiny.pt</code>,
<code>…_small.pt</code>, <code>…_base_plus.pt</code>, <code>…_large.pt</code>) — they
cannot share weights. The architecture is inferred from the filename. If the selected
variant's file isn't on disk, STiM silently falls back to base_plus (no speed/size
change); the checkpoint line under Advanced shows ✓/✗ and offers to download it.</p>

<h3><a name="points_per_batch"></a>points_per_batch <span class="u">(count; memory/speed only)</span></h3>
<p>Number of grid points decoded together in one forward pass. Affects peak memory
and throughput only — never the result. See <a href="#memory">Memory usage</a> for
the formula. Lower to avoid out-of-memory or swapping; raise on a large GPU.</p>

<h3><a name="memory_lowmem"></a>low-memory (1 mask / point) <span class="u">(on/off)</span></h3>
<p>Turns off SAM's <code>multimask_output</code>: the decoder emits 1 mask per point
instead of 3, cutting peak mask memory ~3×. Unlike points_per_batch this DOES touch
the result — it may slightly lower recall on ambiguous sections. Off = SAM default.</p>

<h3><a name="device"></a>device <span class="u">(cpu | cuda | mps)</span></h3>
<p>Where inference runs. Auto resolves CUDA &gt; Apple MPS &gt; CPU. CUDA is fastest;
CPU works anywhere but is much slower (pair with a tiny model and larger sections).</p>
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
