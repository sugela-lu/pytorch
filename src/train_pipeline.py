import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import time
from PIL import Image
import cv2
from pathlib import Path
import torch.amp
import torch.nn.functional as F
import torchvision.transforms as transforms

# Import project modules
from models import SRCNN, ESPCN, EDSR, RCAN, SRResNet
from datasets import SRDataset, MedicalImageDataset, BatchProcessor, RealTimeProcessor
from classical_sr import BicubicInterpolation, IBP, NLMeans, EdgeGuidedSR

class SuperResolutionPipeline:
    """Complete pipeline for image super-resolution training and inference"""
    
    def __init__(
        self,
        data_dir,
        output_dir='results',
        model_type='rcan',
        scale_factor=2,
        batch_size=8,
        num_epochs=100,
        learning_rate=1e-4,
        patch_size=96,
        device=None,
        classical_method=None
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_type = model_type
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patch_size = patch_size
        self.classical_method = classical_method
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initialized Super Resolution Pipeline:")
        print(f"  - Model: {model_type}")
        print(f"  - Scale Factor: {scale_factor}x")
        print(f"  - Device: {self.device}")
        print(f"  - Classical Method: {classical_method if classical_method else 'None'}")
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize classical SR method if specified
        self.classical_sr = self._create_classical_method()
        
        # Initialize loss function
        self.criterion = nn.L1Loss()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize learning rate scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=learning_rate/100)
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = torch.amp.GradScaler('cpu') if self.device.type == 'cpu' else torch.amp.GradScaler('cuda')
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def _create_model(self):
        """Create the specified super-resolution model"""
        if self.model_type.lower() == 'srcnn':
            return SRCNN()
        elif self.model_type.lower() == 'espcn':
            return ESPCN(scale_factor=self.scale_factor)
        elif self.model_type.lower() == 'edsr':
            return EDSR(scale_factor=self.scale_factor, num_blocks=8)
        elif self.model_type.lower() == 'rcan':
            # Use smaller RCAN for faster training
            return RCAN(scale_factor=self.scale_factor, num_groups=5, num_blocks=10)
        elif self.model_type.lower() == 'srresnet':
            return SRResNet(scale_factor=self.scale_factor, num_blocks=8)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_classical_method(self):
        """Create the specified classical SR method"""
        if not self.classical_method:
            return None
            
        if self.classical_method.lower() == 'bicubic':
            return BicubicInterpolation(scale_factor=self.scale_factor)
        elif self.classical_method.lower() == 'ibp':
            return IBP(scale_factor=self.scale_factor, iterations=10)
        elif self.classical_method.lower() == 'nlmeans':
            return NLMeans(scale_factor=self.scale_factor)
        elif self.classical_method.lower() == 'edge':
            return EdgeGuidedSR(scale_factor=self.scale_factor)
        else:
            raise ValueError(f"Unknown classical method: {self.classical_method}")
    
    def _create_dataloaders(self):
        """Create training and validation dataloaders"""
        # Check if we're working with medical images
        is_medical = any(Path(self.data_dir).glob('*.dcm')) or any(Path(self.data_dir).glob('*.nii*'))
        
        # Create appropriate dataset
        if is_medical:
            train_dataset = MedicalImageDataset(
                self.data_dir,
                scale_factor=self.scale_factor,
                patch_size=self.patch_size,
                augment=True,
                split='train'
            )
            val_dataset = MedicalImageDataset(
                self.data_dir,
                scale_factor=self.scale_factor,
                patch_size=self.patch_size,
                augment=False,
                split='val'
            )
        else:
            train_dataset = SRDataset(
                self.data_dir,
                scale_factor=self.scale_factor,
                patch_size=self.patch_size,
                augment=True,
                split='train'
            )
            val_dataset = SRDataset(
                self.data_dir,
                scale_factor=self.scale_factor,
                patch_size=self.patch_size,
                augment=False,
                split='val'
            )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train(self):
        """Train the super-resolution model"""
        print(f"Starting training for {self.num_epochs} epochs...")
        
        # Create dataloaders
        train_loader, val_loader = self._create_dataloaders()
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
            for batch in pbar:
                lr_imgs = batch['lr'].to(self.device)
                hr_imgs = batch['hr'].to(self.device)
                
                self.optimizer.zero_grad()
                
                with torch.amp.autocast('cpu') if self.device.type == 'cpu' else torch.amp.autocast('cuda'):
                    sr_imgs = self.model(lr_imgs)
                    
                    # Ensure output size matches target size
                    if sr_imgs.shape != hr_imgs.shape:
                        sr_imgs = F.interpolate(sr_imgs, size=hr_imgs.shape[-2:], mode='bicubic', align_corners=False)
                    
                    loss = self.criterion(sr_imgs, hr_imgs)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_psnr = 0.0
            
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
                for batch in pbar:
                    lr_imgs = batch['lr'].to(self.device)
                    hr_imgs = batch['hr'].to(self.device)
                    
                    sr_imgs = self.model(lr_imgs)
                    
                    # Ensure output size matches target size
                    if sr_imgs.shape != hr_imgs.shape:
                        sr_imgs = F.interpolate(sr_imgs, size=hr_imgs.shape[-2:], mode='bicubic', align_corners=False)
                    
                    loss = self.criterion(sr_imgs, hr_imgs)
                    
                    # Calculate PSNR
                    mse = nn.MSELoss()(sr_imgs, hr_imgs)
                    psnr = 10 * torch.log10(1.0 / mse)
                    
                    val_loss += loss.item()
                    val_psnr += psnr.item()
                    
                    pbar.set_postfix({'loss': loss.item(), 'psnr': psnr.item()})
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_psnr = val_psnr / len(val_loader)
            self.val_losses.append(avg_val_loss)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.num_epochs} - "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, "
                  f"Val PSNR: {avg_val_psnr:.2f} dB, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint if validation loss improved
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self._save_checkpoint(epoch, avg_val_loss, avg_val_psnr)
                print(f"Saved checkpoint (improved validation loss: {avg_val_loss:.6f})")
            
            # Save sample images every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._save_samples(epoch, val_loader)
        
        # Save final model
        self._save_checkpoint(self.num_epochs - 1, avg_val_loss, avg_val_psnr, is_final=True)
        
        # Plot training history
        self._plot_training_history()
        
        print("Training completed!")
        return self.model
    
    def _save_checkpoint(self, epoch, val_loss, val_psnr, is_final=False):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.output_dir, 
            'checkpoints', 
            f"{self.model_type}_x{self.scale_factor}_{'final' if is_final else 'best'}.pth"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'model_type': self.model_type,
            'scale_factor': self.scale_factor,
        }, checkpoint_path)
    
    def _save_samples(self, epoch, val_loader):
        """Save sample super-resolution results"""
        self.model.eval()
        
        # Get a batch of validation images
        batch = next(iter(val_loader))
        lr_imgs = batch['lr'].to(self.device)
        hr_imgs = batch['hr'].to(self.device)
        filenames = batch['filename']
        
        # Generate super-resolution images
        with torch.no_grad():
            sr_imgs = self.model(lr_imgs)
        
        # Save a few sample images
        num_samples = min(4, len(lr_imgs))
        for i in range(num_samples):
            # Create figure with 3 subplots (LR, SR, HR)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Convert tensors to numpy arrays for plotting
            lr_img = lr_imgs[i].squeeze().cpu().numpy()
            sr_img = sr_imgs[i].squeeze().cpu().numpy()
            hr_img = hr_imgs[i].squeeze().cpu().numpy()
            
            # Plot images
            axes[0].imshow(lr_img, cmap='gray')
            axes[0].set_title('Low Resolution')
            axes[0].axis('off')
            
            axes[1].imshow(sr_img, cmap='gray')
            axes[1].set_title(f'Super Resolution ({self.model_type})')
            axes[1].axis('off')
            
            axes[2].imshow(hr_img, cmap='gray')
            axes[2].set_title('High Resolution (Ground Truth)')
            axes[2].axis('off')
            
            # Resize images to match dimensions for metrics calculation
            if sr_img.shape != hr_img.shape:
                sr_img = cv2.resize(sr_img, hr_img.shape[::-1], interpolation=cv2.INTER_CUBIC)
                
            # Add PSNR value to the plot
            mse = ((sr_img - hr_img) ** 2).mean()
            psnr = 10 * np.log10(1.0 / mse)
            plt.suptitle(f"Sample {i+1}: {filenames[i]} - PSNR: {psnr:.2f} dB")
            
            # Save figure
            sample_path = os.path.join(
                self.output_dir,
                'samples',
                f"epoch_{epoch+1}_sample_{i+1}.png"
            )
            plt.savefig(sample_path, bbox_inches='tight')
            plt.close()
    
    def _plot_training_history(self):
        """Plot training and validation loss history"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training History - {self.model_type.upper()} (x{self.scale_factor})')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        history_path = os.path.join(self.output_dir, f"{self.model_type}_training_history.png")
        plt.savefig(history_path, bbox_inches='tight')
        plt.close()
    
    def load_model(self, checkpoint_path=None):
        """Load a trained model from checkpoint"""
        if checkpoint_path is None:
            # Try to find the best checkpoint
            checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                               if f.startswith(f"{self.model_type}_x{self.scale_factor}")]
            
            if not checkpoint_files:
                print("No checkpoint found. Using untrained model.")
                return self.model
            
            # Prefer final checkpoint, then best checkpoint
            if f"{self.model_type}_x{self.scale_factor}_final.pth" in checkpoint_files:
                checkpoint_path = os.path.join(checkpoint_dir, f"{self.model_type}_x{self.scale_factor}_final.pth")
            else:
                checkpoint_path = os.path.join(checkpoint_dir, f"{self.model_type}_x{self.scale_factor}_best.pth")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {checkpoint_path}")
            print(f"  - Epoch: {checkpoint['epoch'] + 1}")
            print(f"  - Validation Loss: {checkpoint['val_loss']:.6f}")
            if 'val_psnr' in checkpoint:
                print(f"  - Validation PSNR: {checkpoint['val_psnr']:.2f} dB")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
        
        return self.model
    
    def process_image(self, image_path, output_path=None):
        """Process a single image with the trained model"""
        # Load image
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('L')
        elif isinstance(image_path, Image.Image):
            img = image_path.convert('L')
        elif isinstance(image_path, np.ndarray):
            if len(image_path.shape) == 3:
                img = cv2.cvtColor(image_path, cv2.COLOR_RGB2GRAY)
            else:
                img = image_path
            img = Image.fromarray(img)
        else:
            raise ValueError("Unsupported image type")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Process with classical method if specified
        if self.classical_sr:
            classical_result = self.classical_sr.process(img)
            classical_pil = Image.fromarray(classical_result.astype(np.uint8))
        
        # Process with deep learning model
        tensor = torch.from_numpy(np.array(img)).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            output = torch.clamp(output, 0, 1)
        
        # Convert output tensor to PIL image
        output_img = output.squeeze().cpu().numpy() * 255
        output_img = output_img.astype(np.uint8)
        output_pil = Image.fromarray(output_img)
        
        # Save output if path is provided
        if output_path:
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save deep learning result
            output_pil.save(output_path)
            
            # Save classical result if available
            if self.classical_sr:
                classical_path = output_path.replace('.', f'_classical_{self.classical_method}.')
                classical_pil.save(classical_path)
                
                # Create comparison image
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(np.array(img), cmap='gray')
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                axes[1].imshow(classical_result, cmap='gray')
                axes[1].set_title(f'Classical ({self.classical_method})')
                axes[1].axis('off')
                
                axes[2].imshow(output_img, cmap='gray')
                axes[2].set_title(f'Deep Learning ({self.model_type})')
                axes[2].axis('off')
                
                comparison_path = output_path.replace('.', '_comparison.')
                plt.savefig(comparison_path, bbox_inches='tight')
                plt.close()
        
        return output_pil
    
    def create_batch_processor(self, batch_size=4, save_dir=None):
        """Create a batch processor for processing multiple images"""
        return BatchProcessor(
            model=self.model,
            device=self.device,
            batch_size=batch_size,
            scale_factor=self.scale_factor,
            save_dir=save_dir or os.path.join(self.output_dir, 'batch_results')
        )
    
    def create_realtime_processor(self, max_size=512):
        """Create a real-time processor for video or webcam input"""
        return RealTimeProcessor(
            model=self.model,
            device=self.device,
            scale_factor=self.scale_factor,
            max_size=max_size
        )


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Super Resolution Pipeline')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--model', type=str, default='rcan', choices=['srcnn', 'espcn', 'edsr', 'rcan', 'srresnet'], 
                        help='Super-resolution model to use')
    parser.add_argument('--scale', type=int, default=2, help='Scale factor for super-resolution')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patch_size', type=int, default=96, help='Training patch size')
    parser.add_argument('--classical', type=str, choices=['bicubic', 'ibp', 'nlmeans', 'edge'], 
                        help='Classical SR method to compare with')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'batch', 'realtime'],
                        help='Pipeline mode')
    parser.add_argument('--input', type=str, help='Input image or directory for test/batch mode')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = SuperResolutionPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model,
        scale_factor=args.scale,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        patch_size=args.patch_size,
        classical_method=args.classical
    )
    
    # Run pipeline based on mode
    if args.mode == 'train':
        pipeline.train()
    
    elif args.mode == 'test':
        if not args.input:
            print("Error: --input is required for test mode")
            return
        
        # Load model
        pipeline.load_model(args.checkpoint)
        
        # Process single image
        output_path = os.path.join(args.output_dir, f"sr_{os.path.basename(args.input)}")
        result = pipeline.process_image(args.input, output_path)
        print(f"Processed image saved to {output_path}")
    
    elif args.mode == 'batch':
        if not args.input:
            print("Error: --input is required for batch mode")
            return
        
        # Load model
        pipeline.load_model(args.checkpoint)
        
        # Create batch processor
        batch_processor = pipeline.create_batch_processor()
        
        # Process directory
        output_dir = os.path.join(args.output_dir, 'batch_results')
        results = batch_processor.process_directory(args.input, output_dir)
        
        print(f"Processed {len(results)} images. Results saved to {output_dir}")
    
    elif args.mode == 'realtime':
        # Load model
        pipeline.load_model(args.checkpoint)
        
        # Create real-time processor
        realtime_processor = pipeline.create_realtime_processor()
        
        if args.input and os.path.isfile(args.input):
            # Process video file
            output_path = os.path.join(args.output_dir, f"sr_{os.path.basename(args.input)}")
            result_path = realtime_processor.process_video(args.input, output_path)
            print(f"Processed video saved to {result_path}")
        else:
            # Process webcam feed
            print("Starting webcam feed (press 'q' to quit)...")
            
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                start_time = time.time()
                sr_frame = realtime_processor.process_frame(gray_frame)
                process_time = time.time() - start_time
                
                # Add processing time to frame
                fps = 1.0 / process_time
                cv2.putText(sr_frame, f"FPS: {fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display result
                cv2.imshow('Super Resolution (Press q to quit)', sr_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
