from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtCore import Qt, QEvent, QObject
import cv2
import numpy as np
from core.hybrid import create_hybrid_image


class HybridController(QObject):
    def __init__(self, window):
        super().__init__()
        self.window = window

        # Store loaded images
        self.image1 = None
        self.image2 = None

        # ── Style & setup for Image 1 label ──
        self.window.hybrid_label_img1.setCursor(QCursor(Qt.PointingHandCursor))
        self.window.hybrid_label_img1.installEventFilter(self)
        self.window.hybrid_label_img1.setStyleSheet("QLabel { border: 2px dashed #aaa; background-color: #f5f5f5; }")
        self.window.hybrid_label_img1.setScaledContents(False)
        self.window.hybrid_label_img1.setAlignment(Qt.AlignCenter)

        # ── Style & setup for Image 2 label ──
        self.window.hybrid_label_img2.setCursor(QCursor(Qt.PointingHandCursor))
        self.window.hybrid_label_img2.installEventFilter(self)
        self.window.hybrid_label_img2.setStyleSheet("QLabel { border: 2px dashed #aaa; background-color: #f5f5f5; }")
        self.window.hybrid_label_img2.setScaledContents(False)
        self.window.hybrid_label_img2.setAlignment(Qt.AlignCenter)

        # ── Style for output labels ──
        for lbl in [
            self.window.hybrid_label_low_result,
            self.window.hybrid_label_high_result,
            self.window.hybrid_label_hybrid_result,
        ]:
            lbl.setStyleSheet("QLabel { border: 2px solid #aaa; background-color: #f5f5f5; }")
            lbl.setScaledContents(False)
            lbl.setAlignment(Qt.AlignCenter)

        # ── Connect sliders → update value labels ──
        self.window.hybrid_slider_low_cutoff.valueChanged.connect(
            lambda v: self.window.hybrid_label_low_val.setText(str(v))
        )
        self.window.hybrid_slider_high_cutoff.valueChanged.connect(
            lambda v: self.window.hybrid_label_high_val.setText(str(v))
        )
        self.window.hybrid_slider_alpha.valueChanged.connect(
            lambda v: self.window.hybrid_label_alpha_val.setText(f"{v / 100:.2f}")
        )

        # ── Connect Create Hybrid button ──
        self.window.hybrid_btn_create.clicked.connect(self.create_hybrid)

    # ──────────────────────────────────────────────
    #  Double-click detection (same as teammate)
    # ──────────────────────────────────────────────
    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonDblClick:
            if event.button() == Qt.LeftButton:
                if obj == self.window.hybrid_label_img1:
                    self.load_image(1)
                    return True
                elif obj == self.window.hybrid_label_img2:
                    self.load_image(2)
                    return True
        return False

    # ──────────────────────────────────────────────
    #  Load image from file dialog
    # ──────────────────────────────────────────────
    def load_image(self, slot: int):
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Select Image",
            "",
            "Images (*.png *.jpg *.bmp)"
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            return

        if slot == 1:
            self.image1 = img
            self.display_image(img, self.window.hybrid_label_img1)
        else:
            self.image2 = img
            self.display_image(img, self.window.hybrid_label_img2)

    # ──────────────────────────────────────────────
    #  Create hybrid image and show all three results
    # ──────────────────────────────────────────────
    def create_hybrid(self):
        if self.image1 is None or self.image2 is None:
            return  # nothing to do if either image is missing

        low_cutoff  = self.window.hybrid_slider_low_cutoff.value()
        high_cutoff = self.window.hybrid_slider_high_cutoff.value()
        alpha       = self.window.hybrid_slider_alpha.value() / 100.0

        hybrid, low_pass_img, high_pass_img = create_hybrid_image(
            self.image1,
            self.image2,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            alpha=alpha,
        )

        self.display_image(low_pass_img,  self.window.hybrid_label_low_result)
        self.display_image(high_pass_img, self.window.hybrid_label_high_result)
        self.display_image(hybrid,        self.window.hybrid_label_hybrid_result)

    # ──────────────────────────────────────────────
    #  Display helper (same as teammate's pattern)
    # ──────────────────────────────────────────────
    def display_image(self, image: np.ndarray, label):
        # Handle both grayscale and color images
        if len(image.shape) == 2:
            h, w = image.shape
            qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)