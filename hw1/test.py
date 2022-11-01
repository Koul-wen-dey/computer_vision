import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as trans
import time

nepochs = 50
learning = 0.01
PATH = 'vgg19_weight.pt'
BATCH_SIZE = 100
loss_func = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
mean = [x/255 for x in [125.3, 23.0, 113.9]]
std = [x/255 for x in [63.0, 62.1, 66.7]]


def eval(model, loss_func, dataloader):

    model.eval()
    loss, accuracy = 0, 0

    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)

            logits = model(img)
            error = loss_func(logits, label)
            loss += error.item()

            prob, pred_y = logits.data.max(dim=1)
            # accuracy += (pred_y==label.data).float().sum()/label.size(0)
            accuracy += (pred_y==label.data).sum().item()
    loss /= len(dataloader)
    accuracy = accuracy*100.0/(len(dataloader)*BATCH_SIZE)
    return loss, accuracy


def train_epoch(model, loss_func, optimizer, dataloader):

    model.train()
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        error = loss_func(logits, batch_y)
        error.backward()
        optimizer.step()

if __name__ == '__main__':
    train_set = dsets.CIFAR10(root='./data',
                              train=True,
                              download=False,
                              transform=trans.Compose([
                                 trans.RandomHorizontalFlip(),
                                 trans.RandomRotation(30),
                                 trans.RandomCrop(32, padding=4),
                                 trans.ToTensor(),
                                 trans.Normalize(mean, std)
                             ]))
    train_loader = DataLoader(train_set,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0)

    test_set = dsets.CIFAR10(root='./data',
                             train=False,
                             download=False,
                             transform=trans.Compose([
                                trans.ToTensor(),
                                trans.Normalize(mean, std)
                            ]))
    test_loader = DataLoader(test_set,
                         batch_size=BATCH_SIZE,
                         num_workers=0)     

    vgg19 = models.vgg19_bn().to(device)
    vgg19.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    vgg19.classifier[0] = nn.Linear(512,4096).to(device)
    vgg19.classifier[6] = nn.Linear(4096,10).to(device)
    # print(vgg19)

    optimizer = torch.optim.SGD(vgg19.parameters(), lr=learning, momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
    learn_history = []

    print('Start training VGG19')

    for epoch in range(nepochs):
        since = time.time()
        train_epoch(vgg19, loss_func, optimizer, train_loader)

        train_loss, train_acc = eval(vgg19, loss_func, train_loader)
        test_loss, test_acc = eval(vgg19, loss_func, test_loader)
        learn_history.append((train_loss, train_acc, test_loss, test_acc))
        now = time.time()
        print('[%3d/%d, %.0f seconds]|\t tr_err: %.1e, tr_acc: %.2f\t |\t te_err: %.1e, te_acc: %.2f'%(
            epoch+1, nepochs, now-since, train_loss, train_acc, test_loss, test_acc))
        torch.save({'model_state_dict':vgg19.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()
                    }, PATH)
    for his in learn_history:
        print(his)
    with open('err_acc.txt','w') as file:
        for i in range(len(learn_history)):
            file.write(str(learn_history[i][0])+',')
            file.write(str(learn_history[i][1].item())+',')
            file.write(str(learn_history[i][2])+',')
            file.write(str(learn_history[i][3].item()))
        file.close()
    
    # vggn = models.vgg19_bn().to(device)
    # vggn.load_state_dict(torch.load(PATH))