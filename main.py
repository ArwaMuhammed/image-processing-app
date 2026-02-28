import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic
from controllers.main_controller import MainController
from controllers.hybrid_controller import HybridController 


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = QMainWindow()
    uic.loadUi("ui/main_window.ui", window)

    controller = MainController(window)
    hybrid_controller = HybridController(window) 

    # Set default tab to Input tab (index 0)
    window.tabWidget.setCurrentIndex(0)

    window.show()
    sys.exit(app.exec_())