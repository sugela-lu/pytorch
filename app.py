import os
import torch
from flask import Flask, request, render_template, jsonify, send_from_directory
from PIL import Image
import io
import base64
import sys
import time
import json
from werkzeug.utils import secure_filename

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import SRCNN, ESPCN, EDSR, RCAN, SRResNet
from src.classical_sr import BicubicInterpolation, IBP, NLMeans, EdgeGuidedSR
from src.datasets import BatchProcessor, RealTimeProcessor
from torchvision.transforms import ToTensor, ToPILImage, Resize
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    'srcnn': {'class': SRCNN, 'checkpoint': 'src/checkpoints/srcnn_best.pth'},
    'espcn': {'class': ESPCN, 'checkpoint': 'src/checkpoints/espcn_best.pth'},
    'edsr': {'class': EDSR, 'checkpoint': 'src/checkpoints/edsr_best.pth'},
    'rcan': {'class': RCAN, 'checkpoint': 'src/checkpoints/rcan_best.pth'},
    'srresnet': {'class': SRResNet, 'checkpoint': 'src/checkpoints/srresnet_best.pth'}
}

# Classical method configurations
CLASSICAL_METHODS = {
    'bicubic': BicubicInterpolation,
    'ibp': IBP,
    'nlmeans': NLMeans,
    'edge': EdgeGuidedSR
}

# Global variables for models
models = {}
classical_methods = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_processor = None
realtime_processor = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Initialize models
def load_models():
    global models, classical_methods, batch_processor
    
    print(f"Loading models on device: {device}")
    
    # Load deep learning models
    for model_name, config in MODEL_CONFIGS.items():
        try:
            # Initialize model
            if model_name in ['espcn', 'edsr', 'rcan', 'srresnet']:
                model = config['class'](scale_factor=2).to(device)
            else:
                model = config['class']().to(device)
            
            # Look for model checkpoint
            if os.path.exists(config['checkpoint']):
                checkpoint = torch.load(config['checkpoint'], map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded {model_name} from {config['checkpoint']}")
            else:
                print(f"No checkpoint found for {model_name}, using untrained model")
            
            model.eval()
            models[model_name] = model
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    # Initialize classical methods
    for method_name, method_class in CLASSICAL_METHODS.items():
        try:
            classical_methods[method_name] = method_class(scale_factor=2)
            print(f"Initialized classical method: {method_name}")
        except Exception as e:
            print(f"Error initializing {method_name}: {e}")
    
    # Initialize batch processor with default model (SRCNN if available, otherwise first available)
    default_model = models.get('srcnn', next(iter(models.values())) if models else None)
    if default_model:
        batch_processor = BatchProcessor(
            model=default_model,
            device=device,
            batch_size=4,
            scale_factor=2,
            save_dir=app.config['RESULTS_FOLDER']
        )
        print("Initialized batch processor")

# Load models at startup
load_models()

def process_image(image, model_name='srcnn', classical_method=None, scale_factor=2):
    """Process image through the super-resolution model and/or classical method"""
    try:
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')

        # Resize if too large
        if max(image.size) > 1024:
            ratio = 1024.0 / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)

        results = {}
        
        # Process with deep learning model if specified
        if model_name and model_name in models:
            # Convert to tensor and normalize
            transform = ToTensor()
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = models[model_name](image_tensor)
                output = torch.clamp(output, 0, 1)
            
            # Convert back to PIL Image
            dl_result = ToPILImage()(output.squeeze(0).cpu())
            results['deep_learning'] = dl_result
        
        # Process with classical method if specified
        if classical_method and classical_method in classical_methods:
            cl_result = classical_methods[classical_method].process(image)
            if isinstance(cl_result, np.ndarray):
                cl_result = Image.fromarray(cl_result.astype(np.uint8))
            results['classical'] = cl_result
        
        return results
    except Exception as e:
        print(f"Error in process_image: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/technology')
def technology():
    return render_template('technology.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Get parameters
        model_name = request.form.get('model', 'srcnn')
        classical_method = request.form.get('classical_method', None)
        if classical_method == 'none':
            classical_method = None
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save original image
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        image.save(original_path)
        
        # Process through model(s)
        results = process_image(image, model_name, classical_method)
        if not results:
            return jsonify({'error': 'Error processing image'}), 500
        
        response_data = {
            'success': True,
            'original': os.path.relpath(original_path, 'static'),
            'timestamp': timestamp,
            'filename': filename
        }
        
        # Save and encode results
        if 'deep_learning' in results:
            dl_path = os.path.join(app.config['RESULTS_FOLDER'], f"{timestamp}_{model_name}_{filename}")
            results['deep_learning'].save(dl_path)
            response_data['deep_learning'] = os.path.relpath(dl_path, 'static')
            
            # Also provide base64 for immediate display
            buffer = io.BytesIO()
            results['deep_learning'].save(buffer, format='PNG')
            buffer.seek(0)
            response_data['deep_learning_base64'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        if 'classical' in results:
            cl_path = os.path.join(app.config['RESULTS_FOLDER'], f"{timestamp}_{classical_method}_{filename}")
            results['classical'].save(cl_path)
            response_data['classical'] = os.path.relpath(cl_path, 'static')
            
            # Also provide base64 for immediate display
            buffer = io.BytesIO()
            results['classical'].save(buffer, format='PNG')
            buffer.seek(0)
            response_data['classical_base64'] = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify(response_data)
    except Exception as e:
        print(f"Error in upload_file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_process():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400
    
    try:
        # Get parameters
        model_name = request.form.get('model', 'srcnn')
        
        # Update batch processor with selected model
        if model_name in models:
            global batch_processor
            batch_processor = BatchProcessor(
                model=models[model_name],
                device=device,
                batch_size=4,
                scale_factor=2,
                save_dir=app.config['RESULTS_FOLDER']
            )
        
        # Process each file
        timestamp = int(time.time())
        batch_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"batch_{timestamp}")
        os.makedirs(batch_dir, exist_ok=True)
        
        processed_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(batch_dir, filename)
                file.save(file_path)
                processed_files.append(file_path)
        
        # Process the batch
        results = batch_processor.process_directory(batch_dir)
        
        return jsonify({
            'success': True,
            'message': f'Processed {len(results)} files',
            'results': results
        })
    except Exception as e:
        print(f"Error in batch_process: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models and methods"""
    return jsonify({
        'deep_learning': list(models.keys()),
        'classical': list(classical_methods.keys())
    })

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
