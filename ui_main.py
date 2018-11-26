# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1142, 821)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(310, 30, 761, 51))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 761, 51))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollAreaMap = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollAreaMap.setGeometry(QtCore.QRect(310, 110, 650, 650))
        self.scrollAreaMap.setWidgetResizable(True)
        self.scrollAreaMap.setObjectName("scrollAreaMap")
        self.scrollAreaMapWidgetContents = QtWidgets.QWidget()
        self.scrollAreaMapWidgetContents.setGeometry(QtCore.QRect(0, 0, 648, 648))
        self.scrollAreaMapWidgetContents.setObjectName("scrollAreaMapWidgetContents")
        self.scrollAreaMap.setWidget(self.scrollAreaMapWidgetContents)
        self.labelInput = QtWidgets.QLabel(self.centralwidget)
        self.labelInput.setGeometry(QtCore.QRect(20, 20, 251, 161))
        self.labelInput.setObjectName("labelInput")
        self.labelZoomed = QtWidgets.QLabel(self.centralwidget)
        self.labelZoomed.setGeometry(QtCore.QRect(20, 214, 251, 251))
        self.labelZoomed.setObjectName("labelZoomed")
        self.labelFiltered = QtWidgets.QLabel(self.centralwidget)
        self.labelFiltered.setGeometry(QtCore.QRect(20, 480, 250, 250))
        self.labelFiltered.setObjectName("labelFiltered")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1142, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.labelInput.setText(_translate("MainWindow", "realtime"))
        self.labelZoomed.setText(_translate("MainWindow", "TextLabel"))
        self.labelFiltered.setText(_translate("MainWindow", "TextLabel"))

