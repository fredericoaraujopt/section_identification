import io
import numpy as np
from PIL import Image

def preprocess_image(file_path, max_file_size_kb=100, step=10, initial_quality=85):
    """
    Resize and compress the image to meet the file size requirement and return the image as a NumPy array.

    Args:
        file_path (str): Path to the input image file.
        max_file_size_kb (int): Maximum file size in kilobytes.
        step (int): Decrease in size percentage with each iteration.
        initial_quality (int): Starting compression quality.

    Returns:
        np.ndarray: Processed image as a NumPy array.
    """
    img = Image.open(file_path)
    if img.mode == 'P':  
        img = img.convert('RGB')
    img_format = 'JPEG'

    quality = initial_quality
    while True:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img_format, quality=quality)
        size_kb = len(img_byte_arr.getvalue()) / 1024

        if size_kb <= max_file_size_kb:
            img_byte_arr.seek(0)
            img_final = Image.open(img_byte_arr)
            print("Image shape:", img_final.size)
            return np.array(img_final)

        if quality <= 10 or img.width < 50 or img.height < 50:
            raise ValueError("Cannot compress the image below the desired size without excessive quality loss or dimension reduction.")

        new_width = int(img.width * ((100 - step) / 100))
        new_height = int(img.height * ((100 - step) / 100))
        img = img.resize((new_width, new_height), Image.LANCZOS)
        quality = max(10, quality - 5)