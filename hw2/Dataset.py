import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms as trans

class ImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        # self.img_dir = img_dir
        self.transform = None
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = read_image(img_path)
        image = trans.Resize((224,224))(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == '__main__':
    p = './dataset.csv'
    a = ImageDataset(p)
    img, l = a.__getitem__(0)
    print(l)