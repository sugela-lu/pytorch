import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision import transforms
import random
import glob
from typing import List, Tuple, Optional, Union, Dict
import cv2

class SRDataset(Dataset):
    """Base Super Resolution Dataset"""
    def __init__(
        self, 
        data_dir: str,
        scale_factor: int = 2,
        patch_size: int = 96,
        augment: bool = True,
        split: str = 'train',
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    ):
        self.data_dir = data_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.augment = augment
        self.split = split
        
        # Find all image files
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob.glob(os.path.join(data_dir, f'*{ext}')))
        
        # Sort for reproducibility
        self.image_files.sort()
        
        # Split dataset if needed
        if split == 'train':
            self.image_files = self.image_files[:int(0.8 * len(self.image_files))]
        elif split == 'val':
            self.image_files = self.image_files[int(0.8 * len(self.image_files)):]
        
        # Basic transforms
        self.to_tensor = transforms.ToTensor()
        
        # Augmentation transforms
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def _get_patch(self, img, patch_size):
        """Extract a random patch from the image"""
        h, w = img.size
        
        # If image is smaller than patch size, pad it
        if h < patch_size or w < patch_size:
            padding = max(0, patch_size - h, patch_size - w)
            img = transforms.Pad(padding)(img)
            h, w = img.size
        
        # Extract random patch
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        
        return img.crop((y, x, y + patch_size, x + patch_size))
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            # Load image
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            
            # Training mode: extract patches
            if self.split == 'train':
                img = self._get_patch(img, self.patch_size)
                
                # Apply augmentations if enabled
                if self.augment:
                    img = self.augment_transforms(img)
            
            # Convert to tensor
            hr_tensor = self.to_tensor(img)
            
            # Create low-resolution version
            lr_size = tuple(dim // self.scale_factor for dim in hr_tensor.shape[-2:])
            lr_tensor = F.resize(hr_tensor, lr_size, interpolation=transforms.InterpolationMode.BICUBIC)
            
            # Resize LR back to HR size for model input
            lr_tensor = F.resize(lr_tensor, hr_tensor.shape[-2:], interpolation=transforms.InterpolationMode.BICUBIC)
            
            return {
                'lr': lr_tensor,
                'hr': hr_tensor,
                'filename': os.path.basename(img_path)
            }
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy sample
            return {
                'lr': torch.zeros(1, self.patch_size, self.patch_size),
                'hr': torch.zeros(1, self.patch_size, self.patch_size),
                'filename': 'error.jpg'
            }


class MedicalImageDataset(SRDataset):
    """Dataset specifically for medical images with additional preprocessing"""
    def __init__(
        self,
        data_dir: str,
        scale_factor: int = 2,
        patch_size: int = 96,
        augment: bool = True,
        split: str = 'train',
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.dcm', '.nii', '.nii.gz'],
        normalize: bool = True
    ):
        super().__init__(data_dir, scale_factor, patch_size, augment, split, extensions)
        self.normalize = normalize
        
    def _preprocess_medical_image(self, img_tensor):
        """Apply medical-specific preprocessing"""
        # Normalize to [0, 1] range if not already
        if self.normalize:
            if torch.min(img_tensor) < 0 or torch.max(img_tensor) > 1:
                img_tensor = (img_tensor - torch.min(img_tensor)) / (torch.max(img_tensor) - torch.min(img_tensor))
        
        # Apply contrast stretching for better feature visibility
        p_low, p_high = torch.quantile(img_tensor, torch.tensor([0.02, 0.98]))
        img_tensor = torch.clamp((img_tensor - p_low) / (p_high - p_low), 0, 1)
        
        return img_tensor
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        # Apply medical-specific preprocessing
        sample['hr'] = self._preprocess_medical_image(sample['hr'])
        sample['lr'] = self._preprocess_medical_image(sample['lr'])
        
        return sample


class BatchProcessor:
    """Handles batch processing of images for super-resolution"""
    def __init__(
        self,
        model,
        device,
        batch_size: int = 4,
        scale_factor: int = 2,
        save_dir: Optional[str] = None
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.save_dir = save_dir
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    
    def process_directory(self, input_dir: str, output_dir: Optional[str] = None):
        """Process all images in a directory"""
        if output_dir is None:
            output_dir = self.save_dir or os.path.join(input_dir, 'results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dataset and dataloader
        dataset = SRDataset(input_dir, self.scale_factor, augment=False, split='test')
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        # Process batches
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for batch in dataloader:
                lr_images = batch['lr'].to(self.device)
                filenames = batch['filename']
                
                # Process through model
                sr_images = self.model(lr_images)
                
                # Save results
                for i, (sr_img, filename) in enumerate(zip(sr_images, filenames)):
                    # Convert to PIL image
                    sr_pil = transforms.ToPILImage()(sr_img.cpu())
                    
                    # Save to output directory
                    output_path = os.path.join(output_dir, filename)
                    sr_pil.save(output_path)
                    
                    results.append({
                        'filename': filename,
                        'path': output_path
                    })
        
        return results
    
    def process_batch(self, images: List[Union[str, Image.Image, torch.Tensor]]) -> List[torch.Tensor]:
        """Process a batch of images and return super-resolved versions"""
        processed_images = []
        
        # Convert all images to tensors
        for img in images:
            if isinstance(img, str):
                # Load from path
                img = Image.open(img).convert('L')
                img = transforms.ToTensor()(img)
            elif isinstance(img, Image.Image):
                img = transforms.ToTensor()(img.convert('L'))
            elif isinstance(img, np.ndarray):
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = transforms.ToTensor()(img)
            
            processed_images.append(img)
        
        # Stack into batch
        batch = torch.stack(processed_images).to(self.device)
        
        # Process through model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)
        
        return outputs


class RealTimeProcessor:
    """Handles real-time processing of images for super-resolution"""
    def __init__(
        self,
        model,
        device,
        scale_factor: int = 2,
        max_size: int = 512
    ):
        self.model = model
        self.device = device
        self.scale_factor = scale_factor
        self.max_size = max_size
        
        # Set model to evaluation mode
        self.model.eval()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for real-time applications"""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray_frame = frame
        
        # Resize if too large
        h, w = gray_frame.shape
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            gray_frame = cv2.resize(gray_frame, new_size, interpolation=cv2.INTER_AREA)
        
        # Convert to tensor
        frame_tensor = transforms.ToTensor()(gray_frame).unsqueeze(0).to(self.device)
        
        # Process through model
        with torch.no_grad():
            output = self.model(frame_tensor)
            output = torch.clamp(output, 0, 1)
        
        # Convert back to numpy array
        result = output.squeeze().cpu().numpy() * 255
        result = result.astype(np.uint8)
        
        return result
    
    def process_video(self, input_path: str, output_path: str, fps: int = 30):
        """Process a video file"""
        # Open video
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
        
        # Process each frame
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame = self.process_frame(gray_frame)
            
            # Write to output
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        # Release resources
        cap.release()
        out.release()
        
        return output_path


class LIDCDataset(Dataset):
    """LIDC-IDRI Dataset for lung nodules"""
    def __init__(self, root_dir, transform=None, scale_factor=2, cache_data=True):
        self.root_dir = root_dir
        self.transform = transform
        self.scale_factor = scale_factor
        self.to_tensor = transforms.ToTensor()
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
        lr_image = F.resize(hr_image, lr_size, interpolation=transforms.InterpolationMode.BICUBIC)
        lr_image = F.resize(lr_image, hr_image.shape[-2:], interpolation=transforms.InterpolationMode.BICUBIC)
        
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
        self.to_tensor = transforms.ToTensor()
        
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
        lr_image = F.resize(hr_image, lr_size, interpolation=transforms.InterpolationMode.BICUBIC)
        lr_image = F.resize(lr_image, hr_image.shape[-2:], interpolation=transforms.InterpolationMode.BICUBIC)
        
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
        self.to_tensor = transforms.ToTensor()
        
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
        lr_image = F.resize(hr_image, lr_size, interpolation=transforms.InterpolationMode.BICUBIC)
        lr_image = F.resize(lr_image, hr_image.shape[-2:], interpolation=transforms.InterpolationMode.BICUBIC)
        
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
        lr_image = F.resize(hr_image.unsqueeze(0), lr_size, interpolation=transforms.InterpolationMode.BICUBIC).squeeze(0)
        lr_image = F.resize(lr_image.unsqueeze(0), hr_image.shape[-2:], interpolation=transforms.InterpolationMode.BICUBIC).squeeze(0)
        
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
