import os
import torch
from torchvision import datasets
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images_dir = os.path.join(self.data_dir, 'images')
        self.labels_dir = os.path.join(self.data_dir, 'labels')
        self.image_names = os.listdir(self.images_dir)
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.image_names[index])
        label_path = os.path.join(self.labels_dir, self.image_names[index].split('.')[0] + '.txt')
        
        image = Image.open(image_path).convert('RGB')
        with open(label_path, 'r') as f:
            label = f.read().strip()
        
        # Apply any preprocessing to the image and label here, 
        # such as resizing or normalization        
        return image, label


def imageFolderDataloader(data_dir,transform=None, batch_size=4, shuffle=True, num_workers=2):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader
