import torch
import torch.nn.functional as F
import math

def calculate_psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, window_size=11, window=None, size_average=True, full=False):
    """Calculate SSIM (Structural Similarity Index)"""
    K1 = 0.01
    K2 = 0.03
    
    if len(img1.shape) == 2:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 2:
        img2 = img2.unsqueeze(0)
    
    L = 1  # dynamic range
    
    pad = window_size // 2
    
    try:
        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=pad)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=pad)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=pad) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=pad) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=pad) - mu1_mu2
        
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)
        
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
        
        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
        
        if full:
            return ret, cs
        return ret.item()
    
    except Exception as e:
        print(f"Error calculating SSIM: {str(e)}")
        return 0.0
