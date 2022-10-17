from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
import main_ui
import os
import sys
import Q1
import Q2


class GUI(QMainWindow, main_ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.folder = ''
        # self.subL = sub_window()
        # self.subR = sub_window()
        self.pushButton.clicked.connect(self.open_dir)
        self.pushButton_2.clicked.connect(self.open_dir)
        self.pushButton_3.clicked.connect(self.open_dir)

        # Q1

        self.pushButton_5.clicked.connect(
            lambda: Q1.find_corner(self.folder))
        self.pushButton_6.clicked.connect(
            lambda: Q1.find_intrinsic(self.folder))

        self.pushButton_7.clicked.connect(lambda: Q1.find_extrinsic(
            self.folder, int(self.comboBox.currentText())))

        self.pushButton_8.clicked.connect(Q1.find_distortion)
        self.pushButton_9.clicked.connect(lambda: Q1.show_result(self.folder))
        
        # Q2
        self.pushButton_11.clicked.connect(lambda: Q2.on_board(self.folder,self.textEdit.toPlainText()))
        self.pushButton_12.clicked.connect(lambda: Q2.vertical(self.folder,self.textEdit.toPlainText()))

        # Q3
        # self.pushButton_4.clicked.connect()


    def open_dir(self):
        self.folder = QFileDialog.getExistingDirectory(
            self, '開啟資料夾', os.getcwd())


class sub_window(QWidget):
    def __init__(self):
        super(sub_window, self).__init__()
        self.resize(1280, 720)

        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 1280, 720)
        self.label.setText('img')

    def load_img(self):
        directory, _ = QFileDialog.getOpenFileName(
            None, '開啟檔案')
        pic = QPixmap(directory)
        self.label.setPixmap(pic)
        self.label.setScaledContents(True)
        self.show()


if __name__ == '__main__':

    a = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(a.exec_())
