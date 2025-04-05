import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
import nibabel as nib
import pydicom
import SimpleITK as sitk
from pathlib import Path

class LIDCDataset(Dataset):
    """LIDC-IDRI Dataset for lung nodules"""
    def __init__(self, root_dir, transform=None, scale_factor=2, cache_data=True):
        self.root_dir = root_dir
        self.transform = transform
        self.scale_factor = scale_factor
        self.to_tensor = ToTensor()
        self.cache_data = cache_data
        self.cache = {}
        
        # Find all DICOM files
        self.image_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.dcm'):
                    self.image_files.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
            
        # Read DICOM file
        dcm = pydicom.dcmread(self.image_files[idx])
        image = dcm.pixel_array.astype(float)
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        # Convert to PIL Image
        image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Create high-resolution image
        hr_image = self.to_tensor(image)
        
        # Create low-resolution image
        lr_size = tuple(dim // self.scale_factor for dim in hr_image.shape[-2:])
        lr_image = Resize(lr_size)(hr_image)
        lr_image = Resize(hr_image.shape[-2:])(lr_image)
        
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
            
        # Cache the processed images
        if self.cache_data:
            self.cache[idx] = (lr_image, hr_image)
            
        return lr_image, hr_image

class BraTSDataset(Dataset):
    """BraTS Dataset for brain tumor segmentation"""
    def __init__(self, root_dir, transform=None, scale_factor=2, modality='t1'):
        self.root_dir = root_dir
        self.transform = transform
        self.scale_factor = scale_factor
        self.modality = modality
        self.to_tensor = ToTensor()
        
        # Find all NIfTI files for the specified modality
        self.image_files = []
        for path in Path(root_dir).rglob(f'*{modality}.nii.gz'):
            self.image_files.append(str(path))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load NIfTI file
        nifti_img = nib.load(self.image_files[idx])
        image_data = nifti_img.get_fdata()
        
        # Take middle slice for 2D
        middle_slice = image_data[:, :, image_data.shape[2]//2]
        
        # Normalize to [0, 1]
        slice_norm = (middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min())
        
        # Convert to PIL Image
        image = Image.fromarray((slice_norm * 255).astype(np.uint8))
        
        # Create high-resolution image
        hr_image = self.to_tensor(image)
        
        # Create low-resolution image
        lr_size = tuple(dim // self.scale_factor for dim in hr_image.shape[-2:])
        lr_image = Resize(lr_size)(hr_image)
        lr_image = Resize(hr_image.shape[-2:])(lr_image)
        
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        
        return lr_image, hr_image

class COVIDxDataset(Dataset):
    """COVIDx Dataset for COVID-19 chest X-rays"""
    def __init__(self, root_dir, metadata_path, transform=None, scale_factor=2):
        self.root_dir = root_dir
        self.transform = transform
        self.scale_factor = scale_factor
        self.to_tensor = ToTensor()
        
        # Read metadata file
        self.metadata = pd.read_csv(metadata_path)
        self.image_files = self.metadata['filename'].tolist()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Create high-resolution image
        hr_image = self.to_tensor(image)
        
        # Create low-resolution image
        lr_size = tuple(dim // self.scale_factor for dim in hr_image.shape[-2:])
        lr_image = Resize(lr_size)(hr_image)
        lr_image = Resize(hr_image.shape[-2:])(lr_image)
        
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        
        return lr_image, hr_image

class SampleDataset(Dataset):
    """Sample dataset for testing"""
    def __init__(self, root_dir, transform=None, scale_factor=2):
        self.root_dir = root_dir
        self.transform = transform
        self.scale_factor = scale_factor
        
        # Find all tensor files
        self.image_files = []
        for file in os.listdir(root_dir):
            if file.endswith('.pt'):
                self.image_files.append(os.path.join(root_dir, file))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load tensor
        hr_image = torch.load(self.image_files[idx])
        
        # Create low-resolution image
        lr_size = tuple(dim // self.scale_factor for dim in hr_image.shape[-2:])
        lr_image = Resize(lr_size)(hr_image.unsqueeze(0)).squeeze(0)
        lr_image = Resize(hr_image.shape[-2:])(lr_image.unsqueeze(0)).squeeze(0)
        
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        
        return lr_image, hr_image

def get_dataset(dataset_name, root_dir, **kwargs):
    """Factory function to get the appropriate dataset"""
    datasets = {
        'lidc': LIDCDataset,
        'brats': BraTSDataset,
        'covidx': COVIDxDataset,
        'sample': SampleDataset
    }
    
    if dataset_name.lower() not in datasets:
        raise ValueError(f"Dataset {dataset_name} not supported. Available datasets: {list(datasets.keys())}")
    
    return datasets[dataset_name.lower()](root_dir, **kwargs)
