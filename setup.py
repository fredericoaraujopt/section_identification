from setuptools import setup, find_packages

__version__ = "0.2.0"

setup(
    name="section-identification-stim",
    version=__version__,
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pillow",
        "matplotlib",
        "scikit-learn",
        "scikit-image",
        "shapely",
        "opencv-python",
        "tifffile",
        "torch>=2.5.1",
        "torchvision",
        # SAM 2.1 (install from source: SAM2_BUILD_CUDA=0 pip install -e ./sam2)
        "sam2",
        # Zeiss CZI read + in-place metadata edit (edit_czi/set_xml need >=6.0.0)
        "pylibCZIrw>=6.0.0",
        # GUI
        "napari",
        "qtpy",
        "ipywidgets",
        # interactive/ONNX helpers (kept for the legacy/web export paths)
        "onnxruntime",
        "pycocotools",
    ],
    entry_points={
        "console_scripts": [
            "stim=section_identification.interface:main",
        ],
    },
    description=(
        "STiM — Section Targeting interface for Microscopy. A napari-based tool "
        "for detecting tissue sections on (whole-slide) light-microscopy images "
        "with SAM 2.1 and exporting their polygons + fiducials, including a "
        "ZEN-readable CZI for the Shuttle & Find correlative workflow."
    ),
    author="Frederico Araujo",
    author_email="fredrfaa@gmail.com",
    url="https://github.com/fredericoaraujopt/section_identification",
)
