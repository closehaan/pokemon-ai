"""
Semantic Segmentation Backend Server for React App
Uses Mask2Former (ADE20K) for scene understanding + optional matting for clean person masks

Installation:
    pip install flask flask-cors pillow numpy torch torchvision transformers

Usage:
    python segmentation_server.py
"""

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import numpy as np
import os
import threading
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

app = Flask(__name__)
CORS(app)

# Global model variables
segmentation_model = None
processor = None
device = None
model_loaded = False
_model_load_lock = threading.Lock()

# ADE20K label mappings for our categories
# ADE20K has 150 classes - grass (9) is separate from other ground surfaces
LABEL_MAPPING = {
    'sky': [2],  # sky
    'grass': [9],  # grass / lawn only
    'ground': [3, 6, 11, 13, 29, 52, 53],  # floor, road, earth, field, path, sidewalk, runway (not grass)
    'water': [21, 26, 128],  # sea, water, river
    'person': [12],  # person
}


def load_segmentation_model():
    """Load Mask2Former model trained on ADE20K"""
    global segmentation_model, processor, device, model_loaded
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Mask2Former with ADE20K weights (150 semantic classes)
        model_name = "facebook/mask2former-swin-large-ade-semantic"

        print(f"Loading {model_name}...")
        processor = AutoImageProcessor.from_pretrained(model_name)
        segmentation_model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        segmentation_model.to(device)
        segmentation_model.eval()

        print(f"✓ Mask2Former model loaded on {device}")
        print(f"✓ ADE20K dataset with 150 semantic classes")
        print(f"✓ Supports: sky, ground, water, person, and 146 other classes")
        model_loaded = True
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model_loaded = False
        return False


def ensure_model_loaded():
    """
    Render (and similar hosts) expect the web server to bind a port quickly.
    Loading Mask2Former can take a long time (model download + init), so we load
    lazily on first request that needs it.
    """
    global model_loaded
    if model_loaded and segmentation_model is not None and processor is not None:
        return

    with _model_load_lock:
        if model_loaded and segmentation_model is not None and processor is not None:
            return
        ok = load_segmentation_model()
        if not ok:
            raise Exception("Model failed to load (see server logs)")


def map_labels_to_categories(semantic_map):
    """Map ADE20K class IDs to our categories (grass separate from ground)"""
    height, width = semantic_map.shape

    person_mask = np.zeros((height, width), dtype=np.uint8)
    sky_mask = np.zeros((height, width), dtype=np.uint8)
    grass_mask = np.zeros((height, width), dtype=np.uint8)
    ground_mask = np.zeros((height, width), dtype=np.uint8)
    water_mask = np.zeros((height, width), dtype=np.uint8)
    other_mask = np.zeros((height, width), dtype=np.uint8)

    for category, class_ids in LABEL_MAPPING.items():
        for class_id in class_ids:
            mask = (semantic_map == class_id)
            if category == 'person':
                person_mask[mask] = 255
            elif category == 'sky':
                sky_mask[mask] = 255
            elif category == 'grass':
                grass_mask[mask] = 255
            elif category == 'ground':
                ground_mask[mask] = 255
            elif category == 'water':
                water_mask[mask] = 255

    classified = (
        (person_mask > 0)
        | (sky_mask > 0)
        | (grass_mask > 0)
        | (ground_mask > 0)
        | (water_mask > 0)
    )
    other_mask[~classified] = 255

    return person_mask, sky_mask, grass_mask, ground_mask, water_mask, other_mask


def segment_image(image):
    """Segment image using Mask2Former"""
    ensure_model_loaded()

    try:
        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = segmentation_model(**inputs)

        # Post-process to get semantic segmentation map
        # This returns a tensor of shape (height, width) with class IDs
        predicted_semantic_map = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]]
        )[0]

        # Convert to numpy
        semantic_map = predicted_semantic_map.cpu().numpy()

        person_mask, sky_mask, grass_mask, ground_mask, water_mask, other_mask = (
            map_labels_to_categories(semantic_map)
        )

        return person_mask, sky_mask, grass_mask, ground_mask, water_mask, other_mask

    except Exception as e:
        print(f"Segmentation error: {e}")
        import traceback
        traceback.print_exc()
        raise


def create_visualization(person_mask, sky_mask, grass_mask, ground_mask, water_mask, other_mask):
    """Create colored visualization of segmentation (colors must match frontend parsers)"""
    height, width = person_mask.shape
    result = np.zeros((height, width, 3), dtype=np.uint8)

    # Distinct RGB labels for the React client (see App.jsx MASK_BACKEND_COLORS)
    result[person_mask > 0] = [255, 0, 255]      # Magenta - subject
    result[sky_mask > 0] = [135, 206, 235]       # Light sky blue
    result[grass_mask > 0] = [50, 205, 50]       # Lime green - grass
    result[ground_mask > 0] = [139, 69, 19]      # Brown - soil / pavement / non-grass ground
    result[water_mask > 0] = [0, 191, 255]     # Deep sky - water
    result[other_mask > 0] = [169, 169, 169]     # Gray - other

    return Image.fromarray(result)


def calculate_percentages(person_mask, sky_mask, grass_mask, ground_mask, water_mask, other_mask):
    """Calculate percentage of each category in the image"""
    total_pixels = person_mask.size

    person_pct = (np.sum(person_mask > 0) / total_pixels) * 100
    sky_pct = (np.sum(sky_mask > 0) / total_pixels) * 100
    grass_pct = (np.sum(grass_mask > 0) / total_pixels) * 100
    ground_pct = (np.sum(ground_mask > 0) / total_pixels) * 100
    water_pct = (np.sum(water_mask > 0) / total_pixels) * 100
    other_pct = (np.sum(other_mask > 0) / total_pixels) * 100

    return {
        'person': round(person_pct, 2),
        'sky': round(sky_pct, 2),
        'grass': round(grass_pct, 2),
        'ground': round(ground_pct, 2),
        'water': round(water_pct, 2),
        'other': round(other_pct, 2)
    }


@app.route('/segment', methods=['POST'])
def segment():
    """Segment image and return visualization"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Load image
        input_image = Image.open(file.stream).convert('RGB')

        # Segment
        person_mask, sky_mask, grass_mask, ground_mask, water_mask, other_mask = segment_image(
            input_image
        )

        visualization = create_visualization(
            person_mask, sky_mask, grass_mask, ground_mask, water_mask, other_mask
        )

        # Return as PNG
        img_io = BytesIO()
        visualization.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze image and return category percentages as JSON"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Load image
        input_image = Image.open(file.stream).convert('RGB')

        # Segment
        person_mask, sky_mask, grass_mask, ground_mask, water_mask, other_mask = segment_image(
            input_image
        )

        percentages = calculate_percentages(
            person_mask, sky_mask, grass_mask, ground_mask, water_mask, other_mask
        )

        # Return JSON
        return jsonify({
            'status': 'success',
            'categories': percentages
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/remove-background', methods=['POST'])
def remove_background():
    """Remove background, keeping only the person/subject"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Load image
        input_image = Image.open(file.stream).convert('RGB')

        # Segment
        person_mask, sky_mask, grass_mask, ground_mask, water_mask, other_mask = segment_image(
            input_image
        )

        # Create RGBA output with transparent background
        img_array = np.array(input_image)
        rgba = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = img_array
        rgba[:, :, 3] = person_mask  # Alpha channel = person mask

        output_image = Image.fromarray(rgba, 'RGBA')

        # Return as PNG
        img_io = BytesIO()
        output_image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok' if model_loaded else 'starting',
        'service': 'semantic-segmentation',
        'model': 'Mask2Former ADE20K',
        'device': device if device else 'unknown',
        'model_loaded': bool(model_loaded),
        'categories': ['person', 'sky', 'grass', 'ground', 'water', 'other']
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f'\n{"="*60}')
    print(f'Semantic Segmentation Server')
    print(f'{"="*60}')
    print(f'Server: http://localhost:{port}')
    print(f'Model: Mask2Former (ADE20K)')
    print(f'Device: {device}')
    print(f'{"="*60}\n')

    if not model_loaded:
        print('⚠ WARNING: Model not loaded - server may not work properly\n')

    app.run(host='0.0.0.0', port=port, debug=True)