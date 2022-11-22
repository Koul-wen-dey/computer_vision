import torch
from torchvision import models
import torchvision.transforms as trans
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet50().to(device)
resnet.fc = torch.nn.Sequential(
    torch.nn.Linear(2048,1),
    torch.nn.Sigmoid()
).to(device)
check = torch.load('resnet_weight2.pt')
resnet.load_state_dict(check['model_state_dict'])
image = './Dataset_OpenCvDl_Hw2_Q5/validation_dataset/Cat/2.jpg'
image2 = './Dataset_OpenCvDl_Hw2_Q5/inference_dataset/Dog/12052.jpg'
resnet.eval()
with torch.no_grad():
    img = Image.open(image).convert('RGB')
    img2 = Image.open(image2).convert('RGB')
    tf = trans.Compose([
        trans.Resize((224,224)),
        trans.ToTensor()
    ])
    it1 = tf(img2)
    it1 = it1.unsqueeze(0).to(device)
    output = resnet(it1)
    output = torch.round(output)
    print(output)