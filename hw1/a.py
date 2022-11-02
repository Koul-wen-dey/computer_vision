import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as trans
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision

def preprocess(imgs):
    im = torch.from_numpy(imgs)
    im = im.float()  # uint8 to fp16/32
    im /= 255.0
    return im

PATH = 'vgg19_weight.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vggn = models.vgg19_bn().to(device)
vggn.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
vggn.classifier[0] = nn.Linear(512,4096).to(device)
vggn.classifier[6] = nn.Linear(4096,10).to(device)
ch = torch.load(PATH)
vggn.load_state_dict(ch['model_state_dict'])
classes = ('airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
optimizer = torch.optim.SGD(vggn.parameters(), lr=0.01, momentum=0.9, nesterov=True)
optimizer.load_state_dict(ch['optimizer_state_dict'])
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
classes = ('airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
mean = [x/255 for x in [125.3, 23.0, 113.9]]
std = [x/255 for x in [63.0, 62.1, 66.7]]
test_set = dsets.CIFAR10(root='./data',
                             train=False,
                             download=False,
                             transform=trans.Compose([
                                trans.ToTensor(),
                                # trans.Normalize(mean,std)
                            ]))
test_loader = DataLoader(test_set,
                         batch_size=1,
                         num_workers=0)
def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imsave('test_img.png',np.transpose(npimg, (1, 2, 0)))

dataiter = iter(test_loader)
img, la = next(dataiter)
img, la = next(dataiter)
img, la = next(dataiter)
img, la = next(dataiter)
img, la = next(dataiter)
img, la = next(dataiter)
img, la = next(dataiter)
img, la = next(dataiter)
img, la = next(dataiter)
img, la = next(dataiter)

imshow(torchvision.utils.make_grid(img))


vggn.eval()
count = 0
with torch.no_grad():
    # for img, labe in test_loader:
    #     img, labe = img.to(device), labe.to(device)
    #     print(img[0].shape)
    #     logits = vggn(img)
    #     prob, pred_y = logits.data.max(dim=1)
    #     print(labe.data,pred_y)
    #     count+=1
    #     if count > 10:
    #         break

    img = Image.open('test_img.png').convert('RGB')
    # img = np.asarray(img)
    # img = np.transpose(img,(2,0,1))

    tfms = trans.Compose([
        # trans.Resize((32,32)),
        trans.ToTensor(),
        trans.Normalize(mean, std)
        ])
    # img = np.transpose(img)
    # img_tensor = Image.fromarray(img)
    img_tensor = tfms(img).float()
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    # img_tensor = tfms(img_tensor).to(device).unsqueeze(0)
    # print(img_tensor)
    # print(img.size())
    output = vggn(img_tensor)
    probs = torch.nn.functional.softmax(output,dim=1)
    conf, pred = torch.max(probs,1)
    print(conf,classes[pred.item()])
