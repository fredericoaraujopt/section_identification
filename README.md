# STiM: Section Targeting interface for Microscopy

STiM is a napari-based interface for targeting sections on silicon wafers imaged under a microscope, developed to support large-scale connectomics workflows. The tool enables precise targeting of tissue sections for electron microscopy by retrieving their coordinates.

The interface integrates the Segment Anything Model (SAM) for automated section segmentation and includes a manual correction GUI for refining segmentation results.

Results from section targeting experiments on 18 wafers are found in `experiments/image_library`.

## Installation
```
git clone https://github.com/fredericoaraujopt/section_identification.git
cd section_identification
pip install -e .
```

## Tutorial
Once the package is installed, open STiM by running
```
python interface.py
```
📺 **Video demo**: [STiM 🔬](https://www.loom.com/share/48c99d4387db4497963017c24cff7c3b?sid=436aefad-03f1-412b-8d9b-fa19b2b49c7f)
#

Alternatively, to explore the functionality of the package, access the `demo.ipynb` notebook.

1. **Load your image**  
   Indicate the path to the image of interest.

2. **Run automatic detection**  
   Use `automatic_identification()` to identify sections automatically. If you have already run this on the image before, you will retrieve the previously saved masks. Depending on your laptop, running `automatic_identification()` for the first time may take several minutes.

   You can tune several parameters to optimize performance, some of which are:
   - `points_per_side`: Controls the segmentation granularity. Higher values improve detection of smaller sections but increase runtime.
   - `min_mask_area`: Sets a lower threshold for detected mask areas to exclude small, irrelevant objects. Smaller values will segment more granular objects. 
   - `apply_filtering`: If `True`, filters the identified masks by clustering their areas using DBSCAN, then excludes masks that do not belong to the largest cluster. This helps refine segmentation by retaining only the most likely tissue sections, which tend to have similar areas and therefore form the largest cluster.
   - `compress`: If `True`, compresses the input image before segmentation to reduce resource usage. Useful for preliminary testing on low-spec machines, but may degrade segmentation quality.

   These parameters are passed directly into `automatic_identification()` and affect segmentation runtime and accuracy of automatic segmentation of sections. There are more input parameters which affect efficacy of `automatic_identification()`. You are encouraged to experiment with different parameters to identify the most suitable for your applications. Look into `section_detector.py` and `filtering.py`.

3. **Launch manual interface**  
   Call `run_sam_interactive()` to open the manual targeting GUI.

   📺 **Video demo** of the manual targeting process: [Manual targeting 🔬](https://www.loom.com/share/d361c44e708e4592a820a8e2ce8e36a0?sid=fd021e3d-a311-46fb-9ea0-ab85b4af4b5d)
   - Hover over sections to preview masks.
   - **Click** to add a mask.
   - Press **‘r’** to select and remove masks.
   - Press **‘m’** to mark fiducials.
   - Press **‘esc’** to exit the interface.

   Upon exit, the variables `stored_masks`, `new_masks`, and `fiducials` will reflect the latest edits.

5. **Export final coordinates**  
   Run the `export_mask_coordinates()` function to save mask countours and fiducial coordinates into a CSV file.
