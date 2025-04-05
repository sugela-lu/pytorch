# Medical Image Super-Resolution Project

A PyTorch implementation of medical image super-resolution using SRCNN, ESPCN, and EDSR models.

## Features

- Multiple model architectures (SRCNN, ESPCN, EDSR)
- Support for various medical image formats (DICOM, NIfTI, etc.)
- Mixed precision training for faster execution
- Data caching and prefetching for improved performance
- Advanced training features:
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing
  - Gradient clipping
  - Batch normalization
- Model export to ONNX format
- PSNR and SSIM metrics tracking

## Project Structure

```
pytorch/
├── src/
│   ├── models.py      # Model architectures
│   ├── datasets.py    # Dataset implementations
│   └── train.py       # Training pipeline
├── utils/
│   └── metrics.py     # Evaluation metrics
├── data/              # Dataset directory
├── checkpoints/       # Model checkpoints
├── results/          # Training curves
└── models/           # Exported models
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset in the `data` directory

3. Train the model:
```bash
python src/train.py
```

## Performance Optimizations

1. **Data Pipeline**:
   - Implemented data caching to reduce I/O overhead
   - Added proper data prefetching with multiple workers
   - Enabled pin memory for faster GPU transfer

2. **Training**:
   - Mixed precision training (FP16) for faster computation
   - Learning rate scheduling with ReduceLROnPlateau
   - Early stopping to prevent overfitting
   - Gradient clipping for training stability
   - Increased batch size for better GPU utilization

3. **Model**:
   - Added batch normalization for faster convergence
   - Implemented model checkpointing
   - ONNX export for deployment

## Model Selection

- **SRCNN**: Simple and fast, good for initial experiments
- **ESPCN**: Better performance with pixel shuffle upscaling
- **EDSR**: Best quality but slower training

## Metrics

The training script tracks:
- Training/validation loss
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

Results are saved in the `results` directory.
