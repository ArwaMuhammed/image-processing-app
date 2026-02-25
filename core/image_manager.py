import cv2
import numpy as np

class ImageManager:
    def __init__(self):
        self.original_image = None
        self.current_image = None
        self.gray_image = None

    def read_image(self, path):
        """
        Reads an image from the given path and stores it.
        """
        image = cv2.imread(path)  # OpenCV reads BGR by default
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {path}")
        self.original_image = image
        self.current_image = image.copy()
        return self.current_image

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            return self.current_image