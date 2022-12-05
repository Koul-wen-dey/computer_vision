from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage,QPixmap
from PyQt5 import QtWidgets
import torch
from torchsummary import summary
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.datasets as datasets
import torchvision.transforms as trans
import torchvision
import time
import q5_ui
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import cv2
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet50().to(device)
resnet.fc = torch.nn.Sequential(
    torch.nn.Linear(2048,1),
    torch.nn.Sigmoid()
).to(device)


class GUI(QMainWindow,q5_ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.open_img)
        self.pushButton_2.clicked.connect(self.show_img)
        self.pushButton_3.clicked.connect(self.show_distribution)
        self.pushButton_4.clicked.connect(self.show_model)
        self.pushButton_5.clicked.connect(self.show_comparsion)
        self.pushButton_6.clicked.connect(self.show_inference)

    def show_img(self):
        img = Image.open('./Dataset_OpenCvDl_Hw2_Q5/inference_dataset/Cat/8048.jpg')
        img2 = Image.open('./Dataset_OpenCvDl_Hw2_Q5/inference_dataset/Dog/12053.jpg')
        img = trans.Resize((224,224))(img)
        img2 = trans.Resize((224,224))(img2)
        fig = plt.figure(figsize=(1,2))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.imshow(np.asarray(img))
        # plt.figtext(0.1,0.95,'Cat')
        ax2.imshow(np.asarray(img2))
        plt.title('Dog')
        plt.show()
        pass

    def show_distribution(self):
        img = cv2.imread('./fig2.jpg')
        cv2.imshow('Fig 2',img)
        pass
    def show_comparsion(self):
        pass
    def show_inference(self):
        
        pass

    def training_classifier(self):
        
        pass

    def show_model(self):
        global resnet
        summary(resnet,(3,244,244))
        # print(resnet)


    def open_img(self):
        self.image, _ = QFileDialog.getOpenFileName(self,'開啟檔案',os.getcwd())
        print(self.image)
        img = cv2.imread(self.image)
        if img is None:
            print('wrong')
            return
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
        cv2.imshow('',img)
        pass

if __name__ == '__main__':
    a = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(a.exec_())