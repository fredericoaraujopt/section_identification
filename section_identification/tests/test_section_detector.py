import pytest 
from section_identification.section_detector import section_detector
import numpy as np

@pytest.fixture
def make_im():
    image = np.eye(10)
    return image

def test_section_detector(make_im):
    image = make_im
    section_detector(image)