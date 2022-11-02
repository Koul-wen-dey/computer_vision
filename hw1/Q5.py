from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage,QPixmap
from PyQt5 import QtWidgets
import torch
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

PATH = 'vgg19_weight.pt'
batch = 100
epoch = 40
learning_rate = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
mean = [x/255 for x in [125.3, 23.0, 113.9]]
std = [x/255 for x in [63.0, 62.1, 66.7]]
vgg19 = models.vgg19_bn().to(device)
check = torch.load(PATH)
vgg19.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
vgg19.classifier[0] = nn.Linear(512,4096).to(device)
vgg19.classifier[6] = nn.Linear(4096,10).to(device)
vgg19.load_state_dict(check['model_state_dict'])
optimizer = torch.optim.SGD(vgg19.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
optimizer.load_state_dict(check['optimizer_state_dict'])
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)

train_set = datasets.CIFAR10(root='./data',
                              train=True,
                              download=False,
                              transform=trans.Compose([
                                #  trans.RandomHorizontalFlip(),
                                #  trans.RandomRotation(30),
                                #  trans.RandomCrop(32, padding=4),
                                 trans.ToTensor(),
                                #  trans.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                             ]))
train_loader = DataLoader(train_set,
                    batch_size=batch,
                    shuffle=True,
                    num_workers=0)

test_set = datasets.CIFAR10(root='./data',
                             train=False,
                             download=False,
                             transform=trans.Compose([
                                trans.ToTensor(),
                                trans.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                            ]))
test_loader = DataLoader(test_set,
                    batch_size=batch,
                    num_workers=0) 
  
classes = ('airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img,label,index):
    # torchvision.utils.make_grid
    fig = plt.figure(figsize=(3,3))
    for i in range(1,10):
        npimg = torchvision.utils.make_grid(img[index[i-1]]).numpy()
        ax = fig.add_subplot(3,3,i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(classes[label[index[i-1]]])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    tmp = torchvision.utils.make_grid(img[0]).numpy()
    plt.imsave('test_img.png',np.transpose(tmp, (1, 2, 0)))


class GUI(QMainWindow,q5_ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.open_img)
        self.pushButton_2.clicked.connect(self.show_train_img)
        self.pushButton_3.clicked.connect(self.show_model)
        self.pushButton_4.clicked.connect(self.data_augmentation)
        self.pushButton_5.clicked.connect(self.show_accuracy)
        self.pushButton_6.clicked.connect(self.show_inference)

    def show_inference(self):
        global vgg19
        vgg19.eval()
        with torch.no_grad():
            img = Image.open(self.image).convert('RGB')
            tfms = trans.Compose([
                trans.ToTensor(),
                trans.Normalize(mean, std)
                ])
            img_tensor = tfms(img).float()
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)
            output = vgg19(img_tensor)
            probs = torch.nn.functional.softmax(output,dim=1)
            conf, pred = torch.max(probs,1)
            plt.imshow(img)
            print('Confidence:'+str(conf.item()))
            print('Predict Label:'+classes[pred.item()])
            # plt.text(-5, 60, 'Confidence:'+str(conf.item()), fontsize = 15)
            # plt.text(-5, 80, 'Predict Label:'+classes[pred.item()], fontsize = 15)
            plt.show()

            # print(conf,classes[pred.item()])
        pass

    def show_accuracy(self):
        tmp=[]
        tmp2=[]
        with open('err_acc.txt','r') as file:
            t = file.readlines()
            for line in t:
                tmp.append(line.split(','))
        for i in tmp:
            tmp2.append(list(map(lambda x:float(x),i)))
        epo = [i for i in range(1,51)]
        train_loss = [i[0] for i in tmp2]
        train_acc = [i[1] for i in tmp2]
        # test_loss = [i[2] for i in tmp2]
        test_acc = [i[3] for i in tmp2]
        fig, axs = plt.subplots(2)
        axs[0].set_title('Accuracy')
        axs[0].plot(epo,train_acc,label='Training')
        axs[0].plot(epo,test_acc,label='Testing')
        axs[0].legend()
        axs[1].set_title('Loss')
        axs[1].plot(epo,train_loss,label='Training')
        axs[1].legend()
        plt.show()
        pass

    def data_augmentation(self):
        # self.image, _ = QFileDialog.getOpenFileName(self,'開啟檔案',os.getcwd())
        img = cv2.imread(self.image)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
        fig = plt.figure(figsize=(1,3))
        tmp = trans.ToPILImage()(img)
        i1 = trans.RandomRotation(60)(tmp)
        i2 = trans.RandomHorizontalFlip()(tmp)
        i3 = trans.RandomCrop(32, padding=4)(tmp)
        i1 = np.asarray(i1)
        i2 = np.asarray(i2)
        i3 = np.asarray(i3)
        ax1 = fig.add_subplot(1,3,1)
        ax1.imshow(i1)
        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow(i2)
        ax3 = fig.add_subplot(1,3,3)
        ax3.imshow(i3)
        plt.show()
        
        pass

    def training_classifier(self):
        
        pass

    def show_model(self):
        global vgg19
        print(vgg19)

    def show_train_img(self):
        global train_loader,classes
        index_list = []
        for img, label in train_loader:
            for i in range(9):
                index = (i + 10 * random.randint(1,50) )% len(img)
                index_list.append(index)
            imshow(img,label,index_list)
            break        
        pass

    def open_img(self):
        self.image, _ = QFileDialog.getOpenFileName(self,'開啟檔案',os.getcwd())
        img = cv2.imread(self.image)
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
        cv2.imshow('',img)
        pass

if __name__ == '__main__':
    a = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(a.exec_())