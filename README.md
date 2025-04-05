# Medical Image Super-Resolution Project

A comprehensive PyTorch implementation of medical image super-resolution using both classical methods and deep learning models. This project implements a complete image processing pipeline with batch processing and real-time image reconstruction capabilities.

## Features

- **Multiple Deep Learning Models**:
  - SRCNN (Super-Resolution CNN)
  - ESPCN (Efficient Sub-Pixel CNN)
  - EDSR (Enhanced Deep SR)
  - RCAN (Residual Channel Attention Network)
  - SRResNet (Residual Network for SR)

- **Classical Super-Resolution Methods**:
  - Bicubic Interpolation
  - Iterative Back Projection (IBP)
  - Non-Local Means (NLMeans)
  - Edge-Guided Super-Resolution

- **Complete Processing Pipeline**:
  - Single image processing
  - Batch processing for multiple images
  - Real-time image reconstruction
  - Comparison between classical and deep learning methods

- **Web Interface**:
  - Interactive Flask web application
  - Model selection and comparison
  - Batch upload and processing
  - Results visualization and download

- **Advanced Training Features**:
  - Mixed precision training for faster execution
  - Learning rate scheduling
  - Model checkpointing
  - Gradient clipping
  - Batch normalization

- **Support for Medical Images**:
  - Various medical image formats (DICOM, NIfTI, JPEG, PNG)
  - Medical-specific preprocessing
  - Contrast enhancement for better visibility

- **Performance Metrics**:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - Visual comparison tools

## Project Structure

```
pytorch/
├── src/
│   ├── models.py          # Deep learning model architectures
│   ├── classical_sr.py    # Classical super-resolution methods
│   ├── datasets.py        # Dataset implementations with batch processing
│   ├── train.py           # Basic training script
│   ├── train_pipeline.py  # Comprehensive training and inference pipeline
│   └── checkpoints/       # Model checkpoint directory
├── app.py                 # Flask web application
├── templates/             # HTML templates for web interface
│   ├── index.html         # Main page with upload and processing
│   ├── about.html         # About page
│   └── technology.html    # Technical details page
├── static/                # Static assets for web interface
│   ├── uploads/           # Uploaded images
│   └── results/           # Processed results
├── data/                  # Dataset directory
├── utils/                 # Utility functions
├── requirements.txt       # Project dependencies
└── setup.py               # Package setup file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pytorch
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .  # Install package in development mode
```

3. Prepare your dataset in the `data` directory

## Usage

### Web Interface

Run the Flask web application:
```bash
python app.py
```
Then open your browser and navigate to `http://localhost:5000`

### Training Models

Train a specific model:
```bash
python src/train_pipeline.py --data_dir data --model rcan --scale 2 --batch_size 8 --epochs 100
```

### Processing Images

Process a single image:
```bash
python src/train_pipeline.py --mode test --input path/to/image.jpg --model rcan --checkpoint path/to/checkpoint.pth
```

Process a batch of images:
```bash
python src/train_pipeline.py --mode batch --input path/to/image_folder --model rcan --checkpoint path/to/checkpoint.pth
```

Run real-time processing:
```bash
python src/train_pipeline.py --mode realtime --model srcnn --checkpoint path/to/checkpoint.pth
```

## Model Selection Guide

- **SRCNN**: Fast and lightweight model for basic super-resolution. Good starting point.
- **ESPCN**: Efficient model with pixel shuffling for real-time applications.
- **EDSR**: Enhanced Deep SR with residual blocks for better detail preservation.
- **RCAN**: Advanced model with channel attention for capturing fine details. Best quality but slower.
- **SRResNet**: Residual network architecture with skip connections. Good balance of quality and speed.

## Classical Methods

- **Bicubic Interpolation**: Basic interpolation method, fast but limited quality.
- **Iterative Back Projection**: Iteratively refines the high-resolution estimate.
- **Non-Local Means**: Preserves details by averaging similar patches.
- **Edge-Guided SR**: Enhances edges for better structure preservation.

## Performance Optimizations

1. **Data Pipeline**:
   - Implemented data caching to reduce I/O overhead
   - Added proper data prefetching with multiple workers
   - Enabled pin memory for faster GPU transfer

2. **Training**:
   - Mixed precision training (FP16) for faster computation
   - Learning rate scheduling with CosineAnnealingLR
   - Gradient clipping for training stability
   - Increased batch size for better GPU utilization

3. **Inference**:
   - Batch processing for multiple images
   - Real-time processing optimizations
   - Model quantization for faster inference

## Metrics

The training pipeline tracks:
- Training/validation loss
- PSNR (Peak Signal-to-Noise Ratio)
- Visual comparisons between methods

Results are saved in the output directory specified during training.

## License

[MIT License](LICENSE)

## Acknowledgements

- PyTorch team for the deep learning framework
- Original authors of the implemented super-resolution algorithms
- Medical imaging community for datasets and evaluation metrics
