from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap
import cv2
from core.image_manager import ImageManager

#btn_load, btn_reset, lbl_img

class MainController:
    def __init__(self, window):
        self.window = window
        self.manager = ImageManager()

        # connect buttons
        self.window.btn_load.clicked.connect(self.load_image)
        self.window.btn_reset.clicked.connect(self.reset_image)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Select Image",
            "",
            "Images (*.png *.jpg *.bmp)"
        )

        if path:
            img = self.manager.read_image(path)
            self.display_image(img)

    def reset_image(self):
        img = self.manager.reset_image()
        if img is not None:
            self.display_image(img)

    def display_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(
            image_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        self.window.lbl_img.setPixmap(QPixmap.fromImage(qimg))
        self.window.lbl_img.setScaledContents(True)