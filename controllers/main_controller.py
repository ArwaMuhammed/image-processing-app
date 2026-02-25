
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtCore import Qt, QEvent, QObject
import cv2
from core.image_manager import ImageManager

#btn_load, btn_reset, lbl_img

class MainController(QObject):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.manager = ImageManager()

        # connect buttons
        self.window.btn_reset.clicked.connect(self.reset_image)
        
        # make InputImage label clickable (double-click)
        self.window.InputImage.setCursor(QCursor(Qt.PointingHandCursor))
        self.window.InputImage.installEventFilter(self)
        self.window.InputImage.setStyleSheet("QLabel { border: 2px dashed #aaa; background-color: #f5f5f5; }")
        self.window.InputImage.setScaledContents(False)
        self.window.InputImage.setAlignment(Qt.AlignCenter)

    def eventFilter(self, obj, event):
        # Check if the event is a double-click on the InputImage label
        if obj == self.window.InputImage and event.type() == QEvent.MouseButtonDblClick:
            if event.button() == Qt.LeftButton:
                self.load_image()
                return True
        return False

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

        pixmap = QPixmap.fromImage(qimg)
        # Scale pixmap to fit label size while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.window.InputImage.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.window.InputImage.setPixmap(scaled_pixmap)