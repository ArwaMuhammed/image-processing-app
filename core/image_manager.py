import cv2

class ImageManager:
    def __init__(self):
        self.original_image = None
        self.current_image = None
        self.gray_image = None

    def read_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {path}")
        self.original_image = image
        self.current_image = image.copy()
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # always compute once on load
        return self.current_image

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            return self.current_image