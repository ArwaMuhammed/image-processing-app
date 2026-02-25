import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic
from controllers.main_controller import MainController


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = QMainWindow()
    uic.loadUi("ui/main_window.ui", window)

    controller = MainController(window)

    window.show()
    sys.exit(app.exec_())