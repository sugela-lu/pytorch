import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize
import os
from models import SRCNN
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class SimpleDataset(Dataset):
    def __init__(self, size=100, image_size=256):
        self.size = size
        self.image_size = image_size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Create synthetic medical-like image
        img = np.ones((self.image_size, self.image_size), dtype=np.float32)
        
        # Add random shapes
        for _ in range(np.random.randint(1, 4)):
            x = np.random.randint(0, self.image_size)
            y = np.random.randint(0, self.image_size)
            rx = np.random.randint(20, 50)
            ry = np.random.randint(20, 50)
            
            # Create ellipse
            y_grid, x_grid = np.ogrid[-y:self.image_size-y, -x:self.image_size-x]
            mask = (x_grid/rx)**2 + (y_grid/ry)**2 <= 1
            img[mask] = np.random.uniform(0, 0.3)
        
        # Convert to tensor
        img_tensor = torch.FloatTensor(img)
        
        # Create low-res version
        lr_size = self.image_size // 2
        lr_img = Resize((lr_size, lr_size))(img_tensor.unsqueeze(0))
        lr_img = Resize((self.image_size, self.image_size))(lr_img)
        
        return lr_img.squeeze(0), img_tensor

def train():
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = SimpleDataset(size=200)  # Increased dataset size
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    num_epochs = 50
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.unsqueeze(1).to(device)  # Add channel dimension
            hr_imgs = hr_imgs.unsqueeze(1).to(device)  # Add channel dimension
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.unsqueeze(1).to(device)
                hr_imgs = hr_imgs.unsqueeze(1).to(device)
                outputs = model(lr_imgs)
                val_loss += criterion(outputs, hr_imgs).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'checkpoints/srcnn_best.pth')
            print(f'Saved best model with validation loss: {best_val_loss:.4f}')

if __name__ == '__main__':
    print("Starting model training...")
    train()
    print("Training completed!")
