# cnn_model.py
import tensorflow as tf
import numpy as np
from PIL import Image

class CNNModel:
    def __init__(self, model_path: str):
        """
        Initialize the CNN model by loading the .h5 file.
        :param model_path: Path to the saved Keras model (.h5)
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)
        # Define class names in the same order as training labels
        self.class_names = ['ALL', 'AML', 'CLL', 'CML', 'Healthy']

    def preprocess_image(self, image: Image.Image, target_size=(224, 224)):
        """
        Preprocess the input image for prediction.
        :param image: PIL Image
        :param target_size: Expected input size for the model
        :return: Preprocessed numpy array
        """
        #image = image.convert("RGB")
        image = image.resize(target_size)
        img_array = np.asarray(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
        return img_array

    def predict(self, image: Image.Image):
        """
        Run prediction and return class name + confidence score.
        """
        processed_img = self.preprocess_image(image)
        preds = self.model.predict(processed_img)[0]  # first row of batch
        class_index = np.argmax(preds)
        confidence = preds[class_index]
        class_name = self.class_names[class_index]
        return class_name, float(confidence)