from setuptools import setup, find_packages

__version__ = "0.0.1"

setup(
    name='section_identification',
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
        'ipyevents',
        'ipython',
        'onnxruntime',
        'opencv-python',
        'pycocotools'
    ],
    description='section identification',
    author='Frederico Araujo',
    author_email='*****',
    url='*****',
)