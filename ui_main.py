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
        self.labelZoomed.setGeometry(QtCore.QRect(20, 240, 251, 251))
        self.labelZoomed.setObjectName("labelZoomed")
        self.labelFiltered = QtWidgets.QLabel(self.centralwidget)
        self.labelFiltered.setGeometry(QtCore.QRect(20, 500, 250, 250))
        self.labelFiltered.setObjectName("labelFiltered")
        self.labelMapName = QtWidgets.QLabel(self.centralwidget)
        self.labelMapName.setGeometry(QtCore.QRect(20, 210, 181, 21))
        self.labelMapName.setObjectName("labelMapName")
        self.comboBoxConv = QtWidgets.QComboBox(self.centralwidget)
        self.comboBoxConv.setGeometry(QtCore.QRect(310, 20, 651, 23))
        self.comboBoxConv.setObjectName("comboBoxConv")
        self.comboBoxFC = QtWidgets.QComboBox(self.centralwidget)
        self.comboBoxFC.setGeometry(QtCore.QRect(310, 60, 651, 23))
        self.comboBoxFC.setObjectName("comboBoxFC")
        self.labelMapNum = QtWidgets.QLabel(self.centralwidget)
        self.labelMapNum.setGeometry(QtCore.QRect(210, 210, 61, 21))
        self.labelMapNum.setText("")
        self.labelMapNum.setObjectName("labelMapNum")
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
        self.labelFiltered.setText(_translate("MainWindow", "Your advertisement here"))
        self.labelMapName.setText(_translate("MainWindow", "Not chosen"))

