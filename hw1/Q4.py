from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import q4_ui
import sys
import os
import cv2

class GUI(QMainWindow,q4_ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.open_imgL)
        self.pushButton_2.clicked.connect(self.open_imgR)
        self.pushButton_3.clicked.connect(self.find_keypoints)
        self.pushButton_4.clicked.connect(self.match_keypoints)

    def find_keypoints(self):
        img = cv2.imread(self.imageL,cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints = sift.detect(gray,None)
        img = cv2.drawKeypoints(gray,keypoints,img)
        cv2.namedWindow('4.1',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('4.1',1280,960)
        cv2.imshow('4.1',img)
        cv2.waitKey(1200)
        # cv2.destroyWindow('4.1')
        pass
    def match_keypoints(self):
        img1 = cv2.imread(self.imageL,cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(self.imageR,cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create()
        key1,des1 = sift.detectAndCompute(img1,None)
        key2,des2 = sift.detectAndCompute(img2,None)
        bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches,key=lambda v:v.distance)
        result = cv2.drawMatches(img1,key1,img2,key2,matches[:80],img2,flags=2)
        cv2.namedWindow('4.2',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('4.2',1280,960)
        cv2.imshow('4.2',result)
        cv2.waitKey(1200)

        pass

    def open_imgL(self):
        self.imageL, _ = QFileDialog.getOpenFileName(self,'開啟檔案',os.getcwd())
    
    def open_imgR(self):
        self.imageR, _ = QFileDialog.getOpenFileName(self,'開啟檔案',os.getcwd())



if __name__ == '__main__':
    a = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(a.exec_())