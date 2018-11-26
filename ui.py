from PyQt5.QtWidgets import QMainWindow, QFileDialog, QDialog, QWidget, \
                            QApplication, QTabWidget, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QDir, QObject, Qt, QThread, QTimer, QEvent
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QGuiApplication, QCursor, QClipboard

import numpy as np
import os
import sys
import datetime
import cv2

from ui_main import Ui_MainWindow


class FeaturesMap(QLabel):
    def __init__(self):
        super().__init__()
        self._rows = 1
        self._cols = 1
        self.raw_idx = 0

    def mousePressEvent(self, event):
        print(event.pos())
        print(self.getRawNumber(event.pos()))

    def setGridSize(self, size):
        assert len(size) == 2
        self._rows, self._cols = size

    def getRawNumber(self, pos):
        cubeWidth = self.width() // self._rows
        cubeHeight = self.height() // self._cols
        cur_row = pos.x() // cubeWidth
        cur_col = pos.y() // cubeHeight
        self.raw_idx = self._rows * cur_col + cur_row
        return self.raw_idx

    def resetIdx(self):
        self.raw_idx = 0

class Ui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.buttons = ['Layer1', 'Layer2', 'Layer3'] * 8
        self.setButtons(self.buttons)

        self.fmap = FeaturesMap()
        self.ui.scrollAreaMap.setWidget(self.fmap)

        self.currentMap = self.buttons[0]

    def setButtons(self, buttons):
        widget = QWidget()
        layout = QHBoxLayout()
        for button in buttons:
            btn = QPushButton(button)
            btn.setFlat(True)
            btn.clicked.connect(self.btnClicked)
            layout.addWidget(btn)
        widget.setLayout(layout)
        self.ui.scrollArea.setWidget(widget)

        self.buttons = list(sorted(buttons))
        self.currentMap = self.buttons[0]

    @pyqtSlot()
    def btnClicked(self):
        self.currentMap = self.sender().text()
        self.fmap.resetIdx()

    def loadMap(self, image, size):
        # img = cv2.resize(image, (self.fmap.width(), self.fmap.height()))
        # print(self.fmap.width(), self.fmap.height())
        img = (1 - image / (np.max(image) - np.min(image))) * 255
        img = img.astype(np.uint8)
        img = cv2.resize(img, (self.fmap.width(), self.fmap.height()),interpolation = cv2.INTER_NEAREST)
        height, width = img.shape
        qImg = QImage(img, width, height, QImage.Format_Grayscale8)
        self.fmap.setPixmap(QPixmap(qImg))
        self.fmap.setGridSize(size)

    def loadRealImage(self, image):
        img = cv2.resize(image, (self.ui.labelInput.width(), self.ui.labelInput.height()))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        self.ui.labelInput.setPixmap(QPixmap(qImg))

    def loadCell(self, image):
        img = (1 - image / (np.max(image) - np.min(image))) * 255
        img = img.astype(np.uint8)
        # img = cv2.resize(img, (self.ui.labelZoomed.width(), self.ui.labelZoomed.height()))
        img = cv2.resize(img, (224, 224),interpolation = cv2.INTER_NEAREST)
        height, width = img.shape
        qImg = QImage(img, width, height, QImage.Format_Grayscale8)
        self.ui.labelZoomed.setPixmap(QPixmap(qImg))


def run_ui():
    app = QApplication(sys.argv)
    ui = Ui()
    # ui.setGeometry(500, 300, 300, 400)
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_ui()
