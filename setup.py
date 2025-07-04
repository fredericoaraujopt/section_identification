from setuptools import setup, find_packages

__version__ = "1.0.0"

setup(
    name='Section Targeting interface for Microscopy (STiM)',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pillow',
        'matplotlib',
        'scikit-learn',
        'segment-anything',
        'ipywidgets',
        'torch',
        'torchvision',
        'opencv-python',
        'ipycanvas',
        'ipyevents'
        'ipython'
        'onnxruntime'
        'opencv_python'
        'pycocotools'
        'opencv-python',
        'ipycanvas',
        'ipyevents',
        'ipython',
        'onnxruntime',
        'opencv-python',
        'pycocotools'
    ],
    description='STiM is a napari-based interface for targeting sections on silicon wafers imaged under a microscope, developed to support large-scale connectomics workflows.',
    author='Frederico Araujo',
    author_email='fredrfaa@gmail.com',
    url='github.com/fredericoaraujopt/section_identification',
)