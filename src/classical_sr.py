import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

class ClassicalSR:
    """Base class for classical super-resolution methods"""
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor
        
    def process(self, img):
        """Process the input image and return the super-resolved result"""
        raise NotImplementedError("Subclasses must implement this method")
        
    @staticmethod
    def load_image(img_path):
        """Load image from path and convert to grayscale numpy array"""
        if isinstance(img_path, str):
            img = Image.open(img_path).convert('L')
            return np.array(img)
        elif isinstance(img_path, Image.Image):
            img = img_path.convert('L')
            return np.array(img)
        elif isinstance(img_path, np.ndarray):
            if len(img_path.shape) == 3:
                return cv2.cvtColor(img_path, cv2.COLOR_RGB2GRAY)
            return img_path
        else:
            raise ValueError("Unsupported image type")
            
    @staticmethod
    def save_result(img, output_path):
        """Save the result image to the specified path"""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        img.save(output_path)
        
    @staticmethod
    def compare_results(original, result, title="Comparison"):
        """Display original and result images side by side"""
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result, cmap='gray')
        plt.title("Super-Resolved")
        plt.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


class BicubicInterpolation(ClassicalSR):
    """Bicubic interpolation super-resolution"""
    def process(self, img):
        img_array = self.load_image(img)
        h, w = img_array.shape
        
        # Create high-resolution dimensions
        hr_h, hr_w = h * self.scale_factor, w * self.scale_factor
        
        # Perform bicubic interpolation
        result = cv2.resize(img_array, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        
        return result


class IBP(ClassicalSR):
    """Iterative Back Projection super-resolution"""
    def __init__(self, scale_factor=2, iterations=10):
        super().__init__(scale_factor)
        self.iterations = iterations
        
    def process(self, img):
        img_array = self.load_image(img)
        h, w = img_array.shape
        
        # Create initial high-resolution estimate using bicubic interpolation
        hr_h, hr_w = h * self.scale_factor, w * self.scale_factor
        hr_estimate = cv2.resize(img_array, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        
        # Define Gaussian kernel for simulating LR image formation
        kernel_size = 5
        kernel = cv2.getGaussianKernel(kernel_size, 1.0)
        kernel = np.outer(kernel, kernel)
        
        # Iterative back-projection
        for _ in range(self.iterations):
            # Simulate LR image from current HR estimate
            simulated_lr = cv2.filter2D(hr_estimate, -1, kernel)
            simulated_lr = cv2.resize(simulated_lr, (w, h), interpolation=cv2.INTER_CUBIC)
            
            # Compute error between original LR and simulated LR
            lr_error = img_array - simulated_lr
            
            # Back-project error to HR space
            error_up = cv2.resize(lr_error, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
            
            # Update HR estimate
            hr_estimate += 0.1 * error_up  # Small step size for stability
            
        return hr_estimate


class NLMeans(ClassicalSR):
    """Non-Local Means super-resolution"""
    def __init__(self, scale_factor=2, h=10, patch_size=7, patch_distance=11):
        super().__init__(scale_factor)
        self.h = h  # Filter strength
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        
    def process(self, img):
        img_array = self.load_image(img)
        
        # First upscale using bicubic interpolation
        h, w = img_array.shape
        hr_h, hr_w = h * self.scale_factor, w * self.scale_factor
        bicubic_result = cv2.resize(img_array, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply Non-Local Means denoising to enhance details
        result = cv2.fastNlMeansDenoising(
            bicubic_result.astype(np.uint8),
            None,
            h=self.h,
            templateWindowSize=self.patch_size,
            searchWindowSize=self.patch_distance
        )
        
        return result


class EdgeGuidedSR(ClassicalSR):
    """Edge-guided super-resolution"""
    def process(self, img):
        img_array = self.load_image(img)
        h, w = img_array.shape
        
        # Create high-resolution dimensions
        hr_h, hr_w = h * self.scale_factor, w * self.scale_factor
        
        # Detect edges in the original image
        edges = cv2.Canny(img_array.astype(np.uint8), 50, 150)
        
        # Upscale the edge map
        edges_hr = cv2.resize(edges, (hr_w, hr_h), interpolation=cv2.INTER_NEAREST)
        
        # Upscale the original image using bicubic interpolation
        bicubic_result = cv2.resize(img_array, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        
        # Create a mask for edge regions (dilate edges for better coverage)
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.dilate(edges_hr, kernel, iterations=1) / 255.0
        
        # Apply sharpening only to edge regions
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(bicubic_result, -1, sharpen_kernel)
        
        # Combine sharpened edges with bicubic result
        result = bicubic_result * (1 - edge_mask) + sharpened * edge_mask
        
        return result.astype(np.uint8)
