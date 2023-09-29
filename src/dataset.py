import os
from PIL import Image
import torch
from torch.utils.data import Dataset

    
class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(self.image_dir)
        self.transforms = transform

    def __getitem__(self, idx):
        image_name = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_name = os.path.join(self.mask_dir, self.image_filenames[idx].replace('.jpg', '.png'))

        image = Image.open(image_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        
        y1 = mask.type(torch.BoolTensor)
        y2 = torch.bitwise_not(y1)
        mask = torch.cat([y2, y1], dim=0)
        
        return image, mask
            
    def __len__(self):
        return len(self.image_filenames)
