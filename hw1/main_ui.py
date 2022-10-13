# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(943, 629)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 120, 131, 201))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(20, 40, 91, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 90, 91, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 140, 91, 31))
        self.pushButton_3.setObjectName("pushButton_3")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(200, 70, 151, 381))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_5.setGeometry(QtCore.QRect(20, 40, 111, 31))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_6.setGeometry(QtCore.QRect(20, 90, 111, 31))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_8.setGeometry(QtCore.QRect(20, 270, 111, 31))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_9.setGeometry(QtCore.QRect(20, 320, 111, 31))
        self.pushButton_9.setObjectName("pushButton_9")
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_5.setGeometry(QtCore.QRect(10, 140, 131, 111))
        self.groupBox_5.setObjectName("groupBox_5")
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_7.setGeometry(QtCore.QRect(10, 70, 111, 31))
        self.pushButton_7.setObjectName("pushButton_7")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_5)
        self.comboBox.setGeometry(QtCore.QRect(20, 30, 91, 22))
        self.comboBox.setObjectName("comboBox")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(370, 70, 171, 381))
        self.groupBox_3.setObjectName("groupBox_3")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit.setGeometry(QtCore.QRect(20, 40, 113, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_11.setGeometry(QtCore.QRect(10, 90, 151, 31))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_12 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_12.setGeometry(QtCore.QRect(10, 140, 151, 31))
        self.pushButton_12.setObjectName("pushButton_12")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(580, 70, 151, 381))
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_4.setGeometry(QtCore.QRect(10, 40, 131, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 943, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "homework 1"))
        self.groupBox.setTitle(_translate("MainWindow", "Load Image"))
        self.pushButton.setText(_translate("MainWindow", "Load Folder"))
        self.pushButton_2.setText(_translate("MainWindow", "Open Image L"))
        self.pushButton_3.setText(_translate("MainWindow", "Open Image R"))
        self.groupBox_2.setTitle(_translate("MainWindow", "1.Calibration"))
        self.pushButton_5.setText(_translate("MainWindow", "1.1 Find Corner"))
        self.pushButton_6.setText(_translate(
            "MainWindow", "1.2 Find Intrinsic"))
        self.pushButton_8.setText(_translate(
            "MainWindow", "1.4 Find Distortion"))
        self.pushButton_9.setText(_translate("MainWindow", "1.5 Show result"))
        self.groupBox_5.setTitle(_translate(
            "MainWindow", "1.3 Find Extrinsic"))
        self.pushButton_7.setText(_translate(
            "MainWindow", "1.3 Find Extrinsic"))
        # self.pushButton_10.setText(_translate("MainWindow", "1.1 Find Corner"))
        self.groupBox_3.setTitle(_translate(
            "MainWindow", "2.Augmented Reality"))
        self.pushButton_11.setText(_translate(
            "MainWindow", "2.1 Show Words on Board"))
        self.pushButton_12.setText(_translate(
            "MainWindow", "2.2 Show Words Vertically"))
        self.groupBox_4.setTitle(_translate(
            "MainWindow", "3.Stereo Disparity Map"))
        self.pushButton_4.setText(_translate(
            "MainWindow", "3.1 Stereo Disparity Map"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())