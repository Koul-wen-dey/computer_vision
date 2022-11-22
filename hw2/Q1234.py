from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import hw2_ui
import sys
import os
import q1, q2

class GUI(QMainWindow,hw2_ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.load_video) # video
        self.pushButton_2.clicked.connect(self.load_img) #img
        self.pushButton_3.clicked.connect(self.load_folder) #folder
        self.pushButton_4.clicked.connect(lambda: q1.background_subtract(self.video)) #background
        self.pushButton_5.clicked.connect(lambda: q2.preprocessing(self.video)) #preprcessing
        # self.pushButton_6.clicked.connect() #tracking
        # self.pushButton_7.clicked.connect() #perspective
        # self.pushButton_8.clicked.connect() #reconstruction
        # self.pushButton_9.clicked.connect() #compute

    def load_video(self):
        self.video, _ = QFileDialog.getOpenFileName(self,'開啟檔案',os.getcwd())
        print(self.video)
        pass

    def load_img(self):
        self.image, _ = QFileDialog.getOpenFileName(self,'開啟檔案',os.getcwd())
        print(self.image)
        pass
    
    def load_folder(self):
        self.folder = QFileDialog.getExistingDirectory(self,'開啟檔案',os.getcwd())
        print(self.folder)
        pass

if __name__ == '__main__':
    a = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(a.exec_())