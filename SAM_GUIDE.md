# Running SAM in STiM — time, memory, and every parameter explained

This guide explains, from scratch, **what the automatic detector actually does**,
**where time and memory go**, and **what every slider in the GUI means**, so you
can set them deliberately instead of guessing. No prior SAM knowledge assumed.

It is written for this machine — an **Apple M4 Pro, 24 GB unified memory, 20-core
GPU (Metal/MPS)** — and uses **measured numbers from real runs** on
`tard_carbon_coat_001.czi` (a 75 945 × 78 229-pixel wafer montage).

---

## 0. TL;DR (read this first)

- The slow, expensive step is **SAM automatic mask generation**. Everything else
  (reading the image, filtering, ordering, exporting) is seconds.
- **Your 65-minute run was not 65 minutes of computation.** The actual compute
  for `crop_n_layers=2, points_per_side=48` is roughly **8–11 minutes**. The rest
  was your Mac **swapping memory to disk** because total memory use (SAM + napari
  + VS Code + browser + macOS) exceeded **24 GB**. When a Mac swaps, *everything*
  gets slow, not just STiM. We measured **12.8 GB already in swap** before a run
  even started.
- **Two knobs control cost the most:**
  - `crop_n_layers` → **time** (each extra layer multiplies the number of image
    tiles SAM processes: 1 → 5 → 21 → 85 tiles).
  - `points_per_batch` → **peak memory** (how many candidate masks are held in
    memory at once). STiM already auto-caps this for your image size.
- **To stay fast and avoid the freeze:** keep `crop_n_layers` ≤ 2, keep `overview
  long side` ≤ 3072, **lower `points_per_batch` to ~8–12**, and **close other
  heavy apps** (browser, VS Code) before a big run. Watch **Activity Monitor →
  Memory → "Swap Used"**: if it climbs, you are thrashing.

### Measured benchmarks (overview long side = 3072, `points_per_batch` auto-capped to 19)

| `crop_n_layers` | tiles | `points_per_side` | raw masks | time (compute) | notes |
|---|---|---|---|---|---|
| 1 | 5  | 32 | 149  | **98 s**  | fast preview |
| 2 | 21 | 32 | 476  | **287 s** (~4.8 min) | good balance |
| 2 | 21 | 48 | ~900 | **~8–11 min** (compute) | *your run; 65 min observed = swapping* |
| 3 | 85 | 48 | 1708 | **~30–40 min** | thorough; only if you must catch the tiniest sections |

Rule of thumb on this machine: **time ≈ 38 s + ~12 s × (number of tiles)** at
`points_per_side=32`; raising `points_per_side` to 48 multiplies the per-tile part
by about 2.25× (because it scales with the *square* of points-per-side).

---

## 1. What "SAM automatic detection" actually does

SAM (Segment Anything Model) is a neural network that, given an image and a
**prompt** (a point or box), outputs a **mask** — the set of pixels belonging to
the object at that prompt. STiM uses **SAM 2.1**.

You are not prompting it by hand for hundreds of sections, so STiM uses
**"automatic mask generation"**, which works like this:

1. Lay a **regular grid of prompt points** over the image (e.g. 32×32 = 1024
   points).
2. Ask SAM "what object is here?" at **every** point.
3. Collect all the resulting masks, throw away the bad/duplicate ones, and return
   what's left.

That "ask at every point" is why it is expensive: hundreds to thousands of
forward passes through the network, each producing a full-image-sized mask that
must be cleaned up.

SAM internally has two parts, which matters for understanding cost:

- **Image encoder** — a big network that "reads" the whole image **once** and
  turns it into a compact feature map. Run-time is fixed per image (or per tile);
  it does **not** depend on how many points you use.
- **Mask decoder** — a small, fast network that turns *one prompt* + the encoder
  features into a mask. It runs **once per prompt point**, so its total cost
  scales with the number of points.

---

## 2. The pipeline, step by step — where time and memory go

When you click **Run Automatic Detection**, this happens in order:

### Step 1 — Read the image (CZI) → **fast, low memory**
A 13 GB CZI is **never** loaded whole. STiM reads a **downscaled overview** from
the file's built-in image pyramid (`overview long side` sets the size). Reading
the 3072-pixel overview takes **< 1 s** and the overview array is small
(3072 × 2982 × 3 bytes ≈ **27 MB**).
*Time: ~0.5 s. Memory: tens of MB.*

### Step 2 — Contrast/format conversion → **fast, low memory**
The 16-bit grayscale image is stretched (ignoring saturated specular spots),
CLAHE-enhanced, and turned into an 8-bit RGB image SAM can read.
*Time: < 1 s. Memory: tens of MB.*

### Step 3 — SAM automatic mask generation → **THE expensive step**
This is ~95 % of the run time and essentially all the memory pressure. Broken
down:

- **(3a) Build + load the model:** ~1–3 s, ~0.4 GB of weights (the
  `hiera_base_plus` checkpoint is 309 MB).
- **(3b) For each tile** (see `crop_n_layers`): run the **image encoder** once
  (~0.5–2 s/tile on MPS), then lay the point grid and run the **mask decoder** in
  **batches** of `points_per_batch` points.
- **(3c) Post-process each batch** — this is the **memory hot spot**. Every mask
  in a batch is **upscaled to the full tile size** and checked for quality. Peak
  transient memory ≈ `points_per_batch × 3 × tile_width × tile_height × 4 bytes`.
  For the full 3072 × 2982 tile with `points_per_batch=19`, that's **≈ 2 GB held
  at once**, repeatedly allocated and freed for every batch.
- **(3d) Deduplicate** overlapping masks across the grid and across tiles (NMS).

*Time: dominated by (tiles × points). Memory: the model (~0.4 GB) + encoder
features + the ~2 GB per-batch post-processing spike (3c).*

> **Why it can freeze your Mac:** that ~2 GB spike, plus napari holding the
> overview and all your section polygons (~1–2 GB), plus macOS and any other open
> apps, can exceed 24 GB. macOS then moves memory to disk ("swap"), and disk is
> ~100× slower than RAM — so the whole machine crawls. We measured **swap grow by
> 3.4 GB during a single short run**, on top of a baseline that was already
> 12.8 GB swapped.

> **A note on Apple "MPS fallback":** SAM 2.1's GPU support on Apple Metal is
> still "preliminary". A few operations aren't implemented on the GPU, so STiM
> sets `PYTORCH_ENABLE_MPS_FALLBACK=1` and those ops quietly run on the **CPU**
> instead. That keeps results correct but makes parts of the run slower and adds
> CPU load — another reason the machine feels busy during a run.

### Step 4 — Filter for sections → **fast**
Shape gate (drop blobs that can't be sections) + adaptive area band (drop debris
much larger/smaller than the typical section) + area clustering. Pure CPU on a
few hundred masks. *Time: < 2 s. Memory: negligible.* This is **re-run every
time** and is **not cached**, so re-filtering is instant.

### Step 5 — Serial ordering (cross-correlation) → **fast**
Crops a thumbnail per section and orders them by visual similarity.
*Time: a few seconds for a few hundred sections.*

### Step 6 — Export → **I/O-bound, no GPU**
CSV + GeoJSON are instant. Writing an **annotated CZI** copies the whole image
file (~12 GB) and then edits only its metadata — so it's limited by **disk
speed** (≈1–3 min to a local SSD; slower to/from a USB drive), not by SAM.

---

## 3. The 24 GB memory budget (why "less is faster")

Everything shares one **24 GB** pool (Apple "unified memory" = CPU and GPU use
the same RAM). During a GUI detection run, the occupants are roughly:

| Consumer | Typical | Notes |
|---|---|---|
| macOS + Finder/menubar | 3–5 GB | always present |
| Your other apps (browser, VS Code, Slack…) | 3–8 GB | **the easiest thing to reduce** |
| napari + the overview + your polygons | 1–2 GB | grows with section count |
| SAM model weights + encoder features | ~1–1.5 GB | fixed per model |
| **SAM per-batch post-processing spike** | **~2 GB** | set by `points_per_batch` |

When the sum exceeds 24 GB, the Mac swaps and everything slows down. **The single
most effective fixes are (a) close other apps and (b) lower `points_per_batch`.**

---

## 4. Every parameter, explained

For each: *what it is → how it works → effect on the detections → effect on time
→ effect on memory → suggested range.*

### `overview long side` (pixels) — default 3072
- **What:** the size STiM downsamples the huge CZI to before detecting. "3072"
  means the longer side of the working image is 3072 pixels.
- **How:** read from the CZI pyramid; everything downstream works at this size,
  then coordinates are scaled back to full resolution on export.
- **Detections:** bigger = more detail, so **small sections are more likely to be
  found and conjoined sections more likely to separate** — but SAM still shrinks
  each tile to ~1024 px internally, so beyond a point you need tiling
  (`crop_n_layers`) rather than a bigger overview.
- **Time:** bigger overview → slower encoder per tile and bigger post-processing
  masks. Roughly scales with area (double the long side ≈ 4× the work).
- **Memory:** the post-processing spike scales with tile area → **doubling this
  roughly quadruples the memory spike.** This is a major memory lever.
- **Use:** 2048 for quick looks, 3072 balanced, 4096 only when you genuinely need
  finer detail and have closed other apps. ⚠️ Changing this requires **reloading
  the image** to take effect.

### `points_per_side` — default 32
- **What:** density of the prompt-point grid. 32 → a 32×32 = **1024-point grid**
  per tile.
- **How:** SAM is asked "what's here?" at each grid point.
- **Detections:** **more points = denser sampling = more sections found and
  fewer missed/merged**, because more points land inside each small section.
  Too few points and small or touching sections get skipped or fused together
  (this is the "not dense enough / conjoined" problem).
- **Time:** scales with the **square** — 48 vs 32 is `(48/32)² = 2.25×` more
  decoder calls.
- **Memory:** no change to the *peak* (that's set by `points_per_batch`), just
  more batches.
- **Use:** 32 normal, 48–64 when sections are tiny/dense or fusing. Diminishing
  returns above ~64.

### `points_per_batch` — default 64 (auto-capped) — **the memory knob**
- **What:** how many of those grid points are processed **together** in one go.
- **How:** purely a scheduling choice — it does **not change the results at all**,
  only how much work (and memory) happens per step.
- **Detections:** **none.** Identical masks regardless of this value.
- **Time:** bigger batches are slightly faster (better GPU utilisation).
- **Memory:** **this sets the ~2 GB post-processing spike.** Peak ≈
  `points_per_batch × 3 × tile_area × 4 bytes`. Halving it roughly halves the
  spike.
- **Auto-cap:** STiM automatically lowers it so the spike stays under a memory
  budget — on your 3072 overview it is capped from 64 down to **19**. You can set
  it **lower still (e.g. 8–12)** to reduce memory pressure and avoid swapping, at
  a small speed cost. **Setting it higher than the cap has no effect** (the cap
  wins).

### `pred_iou_thresh` — default 0.80
- **What:** a confidence cut. SAM predicts, for each mask, how good it thinks the
  mask is (a self-estimated quality score from 0–1, called "predicted IoU"). This
  threshold drops masks below that score.
- **How:** applied while masks are generated.
- **Detections:** **lower = keep more (denser, but more junk); higher = keep only
  confident masks (cleaner, but may drop faint sections).** Lowering to 0.70 is a
  good way to recover faint/low-contrast sections.
- **Time/Memory:** minor.
- **Use:** 0.70–0.85. Lower if you're missing real sections; raise if you're
  getting noise. (Changing this **re-runs** detection — it is part of the cache
  key.)

### `stability_score_thresh` — fixed at ~0.92 (not in the GUI)
- **What:** another quality cut. A mask is "stable" if it barely changes when you
  nudge the cutoff used to turn SAM's soft output into a hard yes/no mask.
- **Detections:** higher = only crisp, well-defined masks; lower = allow fuzzier
  ones. Left at a sensible default.

### `crop_n_layers` — default 1 in GUI — **the time knob & small-section knob**
- **What:** whether (and how aggressively) SAM also runs on **zoomed-in tiles**
  of the image, not just the whole thing.
- **How:** layer 0 = the whole image (1 tile). Each added layer cuts the image
  into a finer grid of overlapping tiles and runs the *entire* point-grid
  detection on **each** tile, then merges results:
  - `0` → **1 tile** (whole image only)
  - `1` → **5 tiles**
  - `2` → **21 tiles**
  - `3` → **85 tiles**
- **Detections:** because each tile is smaller, sections appear **bigger relative
  to SAM's internal 1024-px working size** → **far better at finding tiny
  sections and separating conjoined ones.** This is the main fix for "not dense
  enough".
- **Time:** **multiplies almost everything** by the tile count. Going 1 → 2 is
  ~4× more tiles; 2 → 3 is ~4× again. This is the dominant time cost.
- **Memory:** peak per-batch spike is similar (tiles are processed one at a
  time), but deeper layers process the full-size tile too, so the spike stays.
- **Use:** 1 for previews, 2 for real results, 3 only when you must catch the
  smallest sections and can tolerate 30–40 min.

### `min_mask_region_area` (pixels) — default 20 (GUI uses 10–20)
- **What:** the smallest mask (in pixels, at overview scale) SAM will keep;
  smaller specks and tiny holes are discarded.
- **Detections:** raise it to drop dust/noise; lower it to keep genuinely tiny
  sections. At a 3072 overview a real section is on the order of tens to a few
  hundred pixels, so 10–20 is reasonable.
- **Time/Memory:** negligible.

### `box_nms_thresh` — fixed at 0.7 (not in the GUI)
- **What:** how aggressively overlapping duplicate masks are merged ("non-maximum
  suppression"). If two masks overlap more than this fraction, the lower-scoring
  one is dropped.
- **Detections:** lower = fewer duplicates but risk merging two real adjacent
  sections; higher = keep more overlaps. Left at a sensible default.

### "Filter for sections" (checkbox) — keep it ON
Not a SAM parameter but a post-step: it (1) drops shapes that can't be sections
(too elongated, too concave, or covering most of the image — e.g. the background
disc), (2) keeps only masks within ~0.2–5× the **median** section size (this is
what removes large **debris**), and (3) keeps the dominant size-cluster. It's
instant and re-applied every run, so toggle it freely.

---

## 5. Worked example — your settings, decoded

You ran: **`points_per_side=48`, `points_per_batch=64`, `pred_iou_thresh=0.7`,
`min_mask_region_area=10`, `crop_n_layers=2`, `overview long side=3072`.**

In plain terms, that told STiM:

1. **`overview long side=3072`** — work on a 3072 × 2982 downscale of the wafer.
2. **`crop_n_layers=2`** — detect on the whole image **and** on a 2×2 and a 4×4
   grid of zoomed tiles → **21 tiles total**. *(This is the biggest cost driver.)*
3. **`points_per_side=48`** — on each tile, probe a **48×48 = 2304-point grid**
   → 21 × 2304 ≈ **48,000 prompt points**. *(2.25× more than the default 32.)*
4. **`points_per_batch=64`** — *requested* 64 per batch, but STiM **auto-capped it
   to 19** for this image size (you'd have seen `capping points_per_batch 64 → 19`
   in the Log). Each batch still spikes ~2 GB during post-processing.
5. **`pred_iou_thresh=0.7`** — keep masks SAM is ≥70 % confident in (lenient → more
   candidates, good for faint sections).
6. **`min_mask_region_area=10`** — keep masks down to 10 px (catch tiny sections).

**Why it took ~65 minutes:** the *computation* for this config is about **8–11
minutes** on your GPU (21 tiles × a dense 48² grid). The extra ~55 minutes was
**memory swapping** — with napari, the overview, hundreds of polygons, and your
other apps all resident, the ~2 GB SAM spike pushed total memory past 24 GB and
macOS thrashed to disk (we measured the machine sitting at ~12.8 GB of swap).

**Same quality, much faster:** keep `crop_n_layers=2`, `points_per_side=48`,
`pred_iou_thresh=0.7`, but **set `points_per_batch` to 10**, **close your browser
and other heavy apps**, and watch that "Swap Used" stays flat. That alone should
bring it down toward the ~10-minute compute floor. If you don't need the very
smallest sections, dropping to `points_per_side=32` roughly halves it again.

---

## 6. Recommended recipes

| Goal | overview | crop_n_layers | points_per_side | points_per_batch | Expected time\* |
|---|---|---|---|---|---|
| **Quick preview** | 2048 | 1 | 32 | 12 | ~1 min |
| **Balanced (default)** | 3072 | 2 | 32 | 12 | ~5 min |
| **Dense / separate touching** | 3072 | 2 | 48 | 10 | ~8–11 min |
| **Thorough (tiny sections)** | 3072 | 3 | 48 | 8 | ~30–40 min |

\* Compute time **assuming you are not swapping**. If "Swap Used" in Activity
Monitor is rising, expect multiples of these — fix memory first.

---

## 7. How to avoid the slowdown (checklist)

1. **Close heavy apps** before a big run (browser tabs, VS Code, Slack). This is
   the biggest single win.
2. **Lower `points_per_batch`** to 8–12. It changes *nothing* about the result —
   only memory.
3. **Keep `overview long side` ≤ 3072** and **`crop_n_layers` ≤ 2** unless you
   truly need more; raising either is expensive in both time and memory.
4. **Open Activity Monitor → Memory tab.** Watch **"Swap Used"** and the **memory
   pressure** graph. Green = fine; yellow/red or rising swap = you're thrashing —
   stop and reduce settings.
5. **Re-running with identical settings is instant** — STiM caches the raw masks
   (per image + model + overview size + `points_per_side` + `crop_n_layers` +
   `min_mask_region_area` + `pred_iou_thresh`). Tuning only the **filter** never
   re-runs SAM.
6. **Keep the working CZI on the internal SSD**, not a USB drive — if the drive
   unmounts mid-run you can lose results, and exporting the annotated CZI is much
   faster locally.

---

*Numbers measured on Apple M4 Pro / 24 GB / macOS 15.5 with SAM 2.1
`hiera_base_plus` on `tard_carbon_coat_001.czi`. Times scale with your settings
and how much memory your other apps are using.*
