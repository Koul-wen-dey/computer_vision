import torch
from torch.nn import functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms as trans
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import models
from tqdm import tqdm
from Focal_Loss import FocalLoss


bz = 16
epoch = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet50().to(device)
resnet.fc = torch.nn.Sequential(
    torch.nn.Linear(2048,1),
    torch.nn.Sigmoid()
).to(device)
check = torch.load('resnet_weight_bc.pt')
resnet.load_state_dict(check['model_state_dict'])

transform = trans.Compose([
    trans.Resize((224,224)),
    trans.RandomHorizontalFlip(p=0.3),
    trans.ToTensor()
    ])
learning = 0.001
optimizer = torch.optim.Adam(resnet.parameters(),lr=learning)
scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[10,20],gamma=0.1)
trainset = ImageFolder('./Dataset_OpenCvDl_Hw2_Q5/training_dataset/',transform=transform)
train_loader = DataLoader(trainset,batch_size=bz,shuffle=True,num_workers=0)
# valid_loader = DataLoader(validset,batch_size=bz,shuffle=True,num_workers=0)

if __name__ == '__main__':
    
    for i in range(epoch):
        resnet.train()
        with tqdm(train_loader,unit='batch') as data:
            correct = 0
            for batch_x, batch_y in data:
                data.set_description(f'Epoch{i+1}')
                batch_x,batch_y = batch_x.to(device), batch_y.to(device)
                batch_y = batch_y.unsqueeze(1).float()
                optimizer.zero_grad()
                output = resnet(batch_x)
                error = F.binary_cross_entropy(output,batch_y)
                error.sum().backward()
                optimizer.step()
                output = torch.round(output)
                correct += torch.sum(output==batch_y.data)
            acc = correct * 100.0 / (len(data) * bz)
            print(f'train accuary:{acc}')
            torch.save({'model_state_dict':resnet.state_dict()},'./resnet_weight_bc.pt')
            print('save checkpoint')