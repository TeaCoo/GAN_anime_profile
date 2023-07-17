from GANime import Ui_GANime
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys


class GANimeUI(QMainWindow, Ui_GANime):
    def __init__(self, parent=None):
        super(GANimeUI, self).__init__(parent)
        self.setupUi(self)

def window():
    app = QApplication(sys.argv)
    win = GANimeUI()
    win.show()
    sys.exit(app.exec_())


window()
