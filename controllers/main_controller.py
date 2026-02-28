from PyQt5.QtWidgets import QFileDialog, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtCore import Qt, QEvent, QObject
import cv2
from core.image_manager import ImageManager
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from core.histogram import Histogram
from core.edges import sobel_edge_detection, prewitt_edge_detection, roberts_edge_detection, canny_edge_detection


class MainController(QObject):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.manager = ImageManager()
        self.equalization_image = None

        # Input tab
        self.window.btn_reset.clicked.connect(self.reset_image)
        self.window.btn_convert_gray.clicked.connect(self.convert_to_gray)
        self._setup_label(self.window.InputImage, dashed=True, clickable=True)

        # Edge detection tab
        self.setup_edge_detection_tab()

        # Equalization tab
        self._setup_label(self.window.equalization_input_image, dashed=True, clickable=True)
        self._setup_label(self.window.equalization_output_image)
        self.window.equalization_btn_apply.clicked.connect(self.apply_equalization)

        self.window.tabWidget.currentChanged.connect(self.on_tab_changed)

    # ── Helpers ────────────────────────────────────────────────

    def _setup_label(self, label, dashed=False, clickable=False):
        border = "2px dashed #aaa" if dashed else "2px solid #aaa"
        label.setStyleSheet(f"QLabel {{ border: {border}; background-color: #f5f5f5; }}")
        label.setScaledContents(False)
        label.setAlignment(Qt.AlignCenter)
        if clickable:
            label.setCursor(QCursor(Qt.PointingHandCursor))
            label.installEventFilter(self)

    def _set_canvas(self, widget, figures):
        """Clear widget layout and populate with one or more FigureCanvas items."""
        self._clear_layout(widget)
        if not widget.layout():
            QVBoxLayout(widget)
        for fig in figures:
            widget.layout().addWidget(FigureCanvas(fig))

    def _clear_layout(self, widget):
        if widget.layout():
            while widget.layout().count():
                item = widget.layout().takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    def display_image(self, image, label):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        qimg = QImage(image_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def display_gray_image(self, gray_image, label):
        h, w = gray_image.shape
        qimg = QImage(gray_image.data, w, h, w, QImage.Format_Grayscale8)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ── Histogram helpers ──────────────────────────────────────

    def _show_rgb_histograms(self, image):
        histB, histG, histR = Histogram.computeHistoColored(image)
        cdfB, cdfG, cdfR   = Histogram.compute_cdf_colored(histB, histG, histR)
        self._set_canvas(self.window.InputHistogram, [
            Histogram.plot_colored_histogram(histB, histG, histR, True, False, False),
            Histogram.plot_colored_histogram(histB, histG, histR, False, True, False),
            Histogram.plot_colored_histogram(histB, histG, histR, False, False, True),
        ])
        self._set_canvas(self.window.InputDistribution, [
            Histogram.plot_cdf_colored(cdfB, cdfG, cdfR, True, False, False),
            Histogram.plot_cdf_colored(cdfB, cdfG, cdfR, False, True, False),
            Histogram.plot_cdf_colored(cdfB, cdfG, cdfR, False, False, True),
        ])

    def _show_gray_histograms(self, gray_image):
        hist = Histogram.computeHistoGray(gray_image)
        cdf  = Histogram.compute_cdf_gray(hist)
        self._set_canvas(self.window.InputHistogram,    [Histogram.plot_gray_histogram(hist)])
        self._set_canvas(self.window.InputDistribution, [Histogram.plot_cdf_gray(cdf)])

    def _show_equalization_input_histogram(self, gray_image):
        hist = Histogram.computeHistoGray(gray_image)
        self._set_canvas(self.window.equalization_input_histogram, [Histogram.plot_gray_histogram(hist)])

    # ── Event filter ───────────────────────────────────────────

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
            if obj == self.window.InputImage:
                self.load_image(); return True
            if obj == self.window.equalization_input_image:
                self.load_equalization_image(); return True
        return False

    # ── Input tab ─────────────────────────────────────────────

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self.window, "Select Image", "", "Images (*.png *.jpg *.bmp)")
        if not path:
            return
        self.manager.read_image(path)
        self.display_image(self.manager.current_image, self.window.InputImage)
        self._show_rgb_histograms(self.manager.original_image)

    def reset_image(self):
        img = self.manager.reset_image()
        if img is not None:
            self.display_image(img, self.window.InputImage)
            self._show_rgb_histograms(self.manager.original_image)

    def convert_to_gray(self):
        if self.manager.gray_image is None:
            return
        gray_bgr = cv2.cvtColor(self.manager.gray_image, cv2.COLOR_GRAY2BGR)
        self.manager.current_image = gray_bgr
        self.display_image(gray_bgr, self.window.InputImage)
        self._show_gray_histograms(self.manager.gray_image)

    # ── Edge detection tab ────────────────────────────────────

    def setup_edge_detection_tab(self):
        """Setup edge detection tab with combo box and labels"""
        # Setup edge input image label
        self.window.edge_input_image.setStyleSheet("QLabel { border: 2px solid #aaa; background-color: #f5f5f5; }")
        self.window.edge_input_image.setScaledContents(False)
        self.window.edge_input_image.setAlignment(Qt.AlignCenter)
        
        # Setup edge output image label
        self.window.edge_output_image.setStyleSheet("QLabel { border: 2px solid #aaa; background-color: #f5f5f5; }")
        self.window.edge_output_image.setScaledContents(False)
        self.window.edge_output_image.setAlignment(Qt.AlignCenter)
        
        # Populate combo box with edge detection options
        self.window.edge_combo.addItems(["Select Mask", "Sobel", "Prewitt", "Roberts", "Canny"])
        
        # Connect combo box signal
        self.window.edge_combo.currentIndexChanged.connect(self.apply_edge_detection)

    def apply_edge_detection(self, index):
        """Apply selected edge detection mask"""
        if self.manager.original_image is None:
            return
        
        selection = self.window.edge_combo.currentText()
        
        if selection == "Select Mask":
            self.window.edge_output_image.clear()
            return
        
        gray = self.manager.gray_image
        
        if selection == "Sobel":
            edges = sobel_edge_detection(gray)
        elif selection == "Prewitt":
            edges = prewitt_edge_detection(gray)
        elif selection == "Roberts":
            edges = roberts_edge_detection(gray)
        elif selection == "Canny":
            edges = canny_edge_detection(gray)
        else:
            return
        
        self.display_gray_image(edges, self.window.edge_output_image)

    # ── Equalization tab ──────────────────────────────────────

    def on_tab_changed(self, index):
        if index == 2 and self.manager.original_image is not None:
            self.display_image(self.manager.original_image, self.window.edge_input_image)
        elif index == 4:
            if self.equalization_image is None and self.manager.original_image is not None:
                self.equalization_image = self.manager.gray_image.copy()
            if self.equalization_image is not None:
                self.display_gray_image(self.equalization_image, self.window.equalization_input_image)
                self._show_equalization_input_histogram(self.equalization_image)

    def load_equalization_image(self):
        path, _ = QFileDialog.getOpenFileName(self.window, "Select Image", "", "Images (*.png *.jpg *.bmp *.jpeg)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            return
        self.equalization_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        self.display_gray_image(self.equalization_image, self.window.equalization_input_image)
        self._show_equalization_input_histogram(self.equalization_image)
        self.window.equalization_output_image.clear()
        self._clear_layout(self.window.equalization_output_histogram)

    def apply_equalization(self):
        if self.equalization_image is None:
            return
        equalized = Histogram.equalize_gray(self.equalization_image)
        self.display_gray_image(equalized, self.window.equalization_output_image)
        hist_eq = Histogram.computeHistoGray(equalized)
        self._set_canvas(self.window.equalization_output_histogram, [Histogram.plot_gray_histogram(hist_eq)])