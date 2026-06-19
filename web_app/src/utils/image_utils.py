# image_utils.py
import os
import datetime
import uuid
from PIL import Image
import numpy as np

class ImageProcessor:
    
    def __init__(self, save_dir=os.path.join("web_app", "processed_images")):
        """
        Initialize the processor with a directory to save images.
        """
        self.save_dir = os.path.abspath(save_dir)  # ensures full path
        os.makedirs(self.save_dir, exist_ok=True)

    def resize_and_save(self, image: Image.Image, target_size=(224, 224)):
        """
        Resize the image and save it into the folder with a unique ID.
        Returns the saved file path.
        """
        image = image.convert("RGB")  # ensure 3 channels
        image = image.resize(target_size)

        # Generate unique filename with UUID + timestamp
        unique_id = str(uuid.uuid4())[:8]  # short unique ID
        report_id = f"p_{unique_id}"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}_{report_id}.png"
        save_path = os.path.join(self.save_dir, filename)

        image.save(save_path)
        return save_path,filename,report_id

    def normalize(self, image_or_path:str, target_size=(224, 224)):
        """
        Normalize the image and return as NumPy array.
        Accepts either a PIL Image object or an image file path.
        Pixel values scaled to [0,1].
        """
        # If a path is provided, open the image
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path)
        else:
            image = image_or_path

        image = image.convert("RGB")
        image = image.resize(target_size)
        img_array = np.array(image) / 255.0
        return img_array
