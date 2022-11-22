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
# PATH = 'vgg19_weight.pt'
# batch = 100
# epoch = 40
# learning_rate = 0.01
# print(device)
# mean = [x/255 for x in [125.3, 23.0, 113.9]]
# std = [x/255 for x in [63.0, 62.1, 66.7]]
# vgg19 = models.vgg19_bn().to(device)
# check = torch.load(PATH)
# vgg19.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
# vgg19.classifier[0] = nn.Linear(512,4096).to(device)
# vgg19.classifier[6] = nn.Linear(4096,10).to(device)
# vgg19.load_state_dict(check['model_state_dict'])
# optimizer = torch.optim.SGD(vgg19.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
# optimizer.load_state_dict(check['optimizer_state_dict'])
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)

# train_set = datasets.CIFAR10(root='./data',
#                               train=True,
#                               download=False,
#                               transform=trans.Compose([
#                                 #  trans.RandomHorizontalFlip(),
#                                 #  trans.RandomRotation(30),
#                                 #  trans.RandomCrop(32, padding=4),
#                                  trans.ToTensor(),
#                                 #  trans.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
#                              ]))
# train_loader = DataLoader(train_set,
#                     batch_size=batch,
#                     shuffle=True,
#                     num_workers=0)

# test_set = datasets.CIFAR10(root='./data',
#                              train=False,
#                              download=False,
#                              transform=trans.Compose([
#                                 trans.ToTensor(),
#                                 trans.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
#                             ]))
# test_loader = DataLoader(test_set,
#                     batch_size=batch,
#                     num_workers=0) 


# def imshow(img,label,index):
#     # torchvision.utils.make_grid
#     fig = plt.figure(figsize=(3,3))
#     for i in range(1,10):
#         npimg = torchvision.utils.make_grid(img[index[i-1]]).numpy()
#         ax = fig.add_subplot(3,3,i)
#         ax.imshow(np.transpose(npimg, (1, 2, 0)))
#         ax.set_title(classes[label[index[i-1]]])
#         ax.set_xticks([])
#         ax.set_yticks([])
#     plt.show()
#     tmp = torchvision.utils.make_grid(img[0]).numpy()
#     plt.imsave('test_img.png',np.transpose(tmp, (1, 2, 0)))


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