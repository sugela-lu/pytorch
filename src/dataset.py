import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose
import numpy as np

class MedicalImageDataset(Dataset):
    def __init__(self, data_dir, scale_factor=2, transform=None):
        """
        Args:
            data_dir (str): Directory with medical images
            scale_factor (int): Factor to downscale images for creating LR images
            transform (callable, optional): Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.scale_factor = scale_factor
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Basic transforms
        self.to_tensor = ToTensor()
        self.resize = Resize((224, 224))  # Fixed size for simplicity
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Resize to fixed size
        image = self.resize(image)
        
        # Create high-resolution (HR) image
        hr_image = self.to_tensor(image)
        
        # Create low-resolution (LR) image
        lr_size = tuple(dim // self.scale_factor for dim in hr_image.shape[-2:])
        lr_image = Resize(lr_size)(hr_image)
        lr_image = Resize(hr_image.shape[-2:])(lr_image)  # Upscale back to original size
        
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
            
        return lr_image, hr_image
