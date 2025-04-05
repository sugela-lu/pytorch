import os
import torch
from flask import Flask, request, render_template, jsonify
from PIL import Image
import io
import base64
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import SRCNN
from torchvision.transforms import ToTensor, ToPILImage, Resize
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Initialize model
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    
    # Look for model checkpoint
    checkpoint_path = os.path.join('src', 'checkpoints', 'srcnn_best.pth')
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded trained model from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        print("Using untrained model")
    
    model.eval()
    return model, device

model, device = load_model()

def process_image(image):
    """Process image through the super-resolution model"""
    try:
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize if too large
        if max(image.size) > 512:
            ratio = 512.0 / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)
        
        # Convert to tensor and normalize
        transform = ToTensor()
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            output = torch.clamp(output, 0, 1)
        
        # Convert back to PIL Image
        output_image = ToPILImage()(output.squeeze(0).cpu())
        return output_image
    except Exception as e:
        print(f"Error in process_image: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Process through model
        result_image = process_image(image)
        if result_image is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        # Save result
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
        # Convert to base64 for sending to frontend
        encoded_image = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': encoded_image
        })
    except Exception as e:
        print(f"Error in upload_file: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
