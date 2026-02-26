
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtCore import Qt, QEvent, QObject
import cv2
from core.image_manager import ImageManager
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from core.histogram import Histogram



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
        
        # Setup gray image label
        self.window.GrayImage.setStyleSheet("QLabel { border: 2px solid #aaa; background-color: #f5f5f5; }")
        self.window.GrayImage.setScaledContents(False)
        self.window.GrayImage.setAlignment(Qt.AlignCenter)

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
            # Convert to gray (automatically stored in manager)
            gray_img = self.manager.convertToGray(img)

            # Display both images
            self.display_image(img, self.window.InputImage)
            self.display_gray_image(gray_img, self.window.GrayImage)
        
            # Display histograms
            self.display_histograms()

            self.display_CDF()

#________________________________________________________________________
        
    def reset_image(self):
        img = self.manager.reset_image()
        if img is not None:
            self.display_image(img, self.window.InputImage)
            # Clear gray image and histograms
            self.window.GrayImage.clear()
            self.clear_widget_layout(self.window.InputHistogram)
            self.clear_widget_layout(self.window.GrayHistogram)
            self.clear_widget_layout(self.window.InputDistribution)
            self.clear_widget_layout(self.window.GrayDistribution)

    def display_image(self, image, label):
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
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def display_gray_image(self, gray_image, label):
        h, w = gray_image.shape
        bytes_per_line = w

        qimg = QImage(
            gray_image.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_Grayscale8
        )

        pixmap = QPixmap.fromImage(qimg)
        # Scale pixmap to fit label size while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)



    def display_histograms(self):
        if self.manager.original_image is not None:
            # Calculate histograms
            histB, histG, histR = Histogram.computeHistoColored(self.manager.original_image)
            gray_hist = Histogram.computeHistoGray(self.manager.gray_image)
            
            # Get matplotlib figures
            color_fig = Histogram.plot_colored_histogram(histB, histG, histR)
            gray_fig = Histogram.plot_gray_histogram(gray_hist)
            
            # Display plots
            self.add_plot_to_widget(color_fig, self.window.InputHistogram)
            self.add_plot_to_widget(gray_fig, self.window.GrayHistogram)

    def display_CDF(self):
        if self.manager.original_image is not None:
            # Calculate histograms
            histB, histG, histR = Histogram.computeHistoColored(self.manager.original_image)
            gray_hist = Histogram.computeHistoGray(self.manager.gray_image)
            
            # Calculate CDFs
            cdfB, cdfG, cdfR = Histogram.compute_cdf_colored(histB, histG, histR)
            cdf_gray = Histogram.compute_cdf_gray(gray_hist)
            
            # Get matplotlib figures for CDFs
            cdf_color_fig = Histogram.plot_cdf_colored(cdfB, cdfG, cdfR)
            cdf_gray_fig = Histogram.plot_cdf_gray(cdf_gray)
            
            # Display plots
            self.add_plot_to_widget(cdf_color_fig, self.window.InputDistribution)
            self.add_plot_to_widget(cdf_gray_fig, self.window.GrayDistribution)

    def add_plot_to_widget(self, figure, widget):
        """Helper method to add matplotlib figure to a widget container"""
        # Clear existing widgets
        self.clear_widget_layout(widget)
        
        # Create canvas from figure
        canvas = FigureCanvas(figure)
        
        # Create layout if needed and add canvas
        if not widget.layout():
            QVBoxLayout(widget)
        widget.layout().addWidget(canvas)



    def clear_widget_layout(self, widget):
        """Remove all widgets from a widget's layout"""
        if widget.layout() is not None:
            while widget.layout().count():
                item = widget.layout().takeAt(0)
                if item.widget():
                    item.widget().deleteLater()