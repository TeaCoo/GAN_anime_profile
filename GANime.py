# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GANime.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_GANime(object):
    def setupUi(self, GANime):
        GANime.setObjectName("GANime")
        GANime.resize(1280, 720)
        self.pushButton = QtWidgets.QPushButton(GANime)
        self.pushButton.setGeometry(QtCore.QRect(460, 310, 301, 201))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(GANime)
        QtCore.QMetaObject.connectSlotsByName(GANime)

    def retranslateUi(self, GANime):
        _translate = QtCore.QCoreApplication.translate
        GANime.setWindowTitle(_translate("GANime", "GANime"))
        self.pushButton.setText(_translate("GANime", "PushButton"))
