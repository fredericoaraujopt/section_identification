# Section Targeting

This Python package provides an interface for targeting sections on silicon wafers imaged under a microscope, developed to support large-scale connectomics workflows. The tool enables precise targeting of tissue sections for electron microscopy by retrieving their coordinates.

The interface integrates the Segment Anything Model (SAM) for automated section segmentation and includes a manual correction GUI for refining segmentation results.

## Installation
```
git clone https://github.com/fredericoaraujopt/section_identification.git
cd section_identification
pip install -e .
```

## Tutorial

Use the `demo.ipynb` notebook to explore the functionality of the package. The notebook follows the operation steps:

1. **Load your image**  
   Indicate the path to the image of interest.

2. **Run automatic detection**  
   Use `automatic_identification()` to identify sections automatically. If you have already run this on the image before, you will retrieve the previously saved masks. Depending on your laptop, running `automatic_identification()` for the first time may take several minutes.

   You can tune several parameters to optimize performance, some of which are:
   - `points_per_side`: Controls the segmentation granularity. Higher values improve detection of smaller sections but increase runtime.
   - `min_mask_area`: Sets a lower threshold for detected mask areas to exclude small, irrelevant objects.
   - `apply_filtering`: If `True`, runs DBSCAN clustering to eliminate outlier masks based on size. Helps refine segmentation by keeping only likely tissue sections.
   - `compress`: If `True`, compresses the input image before segmentation to reduce resource usage. Useful for preliminary testing on low-spec machines, but may degrade segmentation quality.

   These parameters are passed directly into `automatic_identification()` to adapt the segmentation pipeline to your image conditions and computing environment.

3. **Launch manual interface**  
   Call `run_sam_interactive()` to open the manual targeting GUI.
   - Hover over sections to preview masks.
   - **Click** to add a mask.
   - Press **‘r’** to select and remove masks.
   - Press **‘m’** to mark fiducials.
   - Press **‘esc’** to exit the interface.

   Upon exit, the variables `stored_masks`, `new_masks`, and `fiducials` will reflect the latest edits.

4. **Export final coordinates**  
   Run the `export_mask_coordinates()` function to save mask countours and fiducial coordinates into a CSV file.

📺 **Video demo** of the manual targeting process: [Manual targeting 🔬](https://www.loom.com/share/d361c44e708e4592a820a8e2ce8e36a0?sid=fd021e3d-a311-46fb-9ea0-ab85b4af4b5d)


# Basic git tutorial
## Contributing new changes
Make sure your main branch is up to date
```
git checkout main
git pull
```
Checkout new branch
```
git checkout -b <name>-<feature>
```
Make changes.
Make sure tests pass:
```
pytest .
```
Push changes:
```
git add <files to be pushed>
git commit -m "<commit message>"
git push # might need to use git push --set-upstream origin <name>-<feature>
```

## Pulling changes
Make sure your main branch is up to date
```
git checkout main
git pull
```
Checkout the branch that you want to update
```
git checkout <name>-<feature>
git merge main
```
