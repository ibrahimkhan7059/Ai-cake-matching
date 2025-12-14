from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import logging
import time
import json

# Configure logging
logging.basicConfig(
    filename='api_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load Keras model (.keras format)
MODEL_PATH = 'best_cake_model_mobilenet.keras'
CLASS_NAMES = ["cheesecake", "chocolate", "extra_cakes", "not_cake", "red_velvet"]

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/api/v1/ai-cake', methods=['POST'])
def predict_cake():
    request_time = time.time()
    logging.info(f"Received prediction request from: {request.remote_addr}")
    logging.info(f"Request headers: {dict(request.headers)}")
    
    if 'image' not in request.files:
        logging.error("No image in request files")
        return jsonify({'success': False, 'message': 'No image uploaded', 'error': 'no_image'}), 400
    
    file = request.files['image']
    logging.info(f"Image filename: {file.filename}, Content type: {file.content_type}")
    
    try:
        image = Image.open(file.stream)
        logging.info(f"Image opened successfully. Size: {image.size}, Mode: {image.mode}")
        
        img = preprocess_image(image)
        logging.info("Image preprocessed successfully")
        
        preds = model.predict(img)
        pred_idx = np.argmax(preds[0])
        confidence = float(preds[0][pred_idx])
        class_probs = {name: float(prob) for name, prob in zip(CLASS_NAMES, preds[0])}
        
        logging.info(f"Prediction results - Class: {CLASS_NAMES[pred_idx]}, Confidence: {confidence}")
        logging.info(f"Class probabilities: {json.dumps(class_probs)}")
        
        result = {
            'category': CLASS_NAMES[pred_idx],
            'confidence': confidence,
            'probabilities': class_probs
        }
        
        # Generate mock matching cakes based on predicted category
        matching_cakes = []
        if CLASS_NAMES[pred_idx] != 'not_cake':
            # Create sample cakes for the predicted category
            category_name = CLASS_NAMES[pred_idx].replace('_', ' ').title()
            for i in range(4):
                matching_cakes.append({
                    'id': i + 1,
                    'name': f'{category_name} Cake {i + 1}',
                    'category': category_name,
                    'price': f'Rs. {2000 + (i * 500)}',
                    'image': f'https://via.placeholder.com/200x200?text={category_name}+{i+1}',
                    'description': f'Delicious {category_name.lower()} cake, perfect for celebrations!'
                })
        
        response = {
            'success': True, 
            'prediction': result,
            'matching_cakes': matching_cakes,
            'is_cake': CLASS_NAMES[pred_idx] != 'not_cake'
        }
        
        if CLASS_NAMES[pred_idx] == 'not_cake':
            response['disclaimer'] = 'Please upload a cake image only.'
            response['message'] = 'This image does not appear to be a cake'
        
        process_time = time.time() - request_time
        logging.info(f"Request processed in {process_time:.2f} seconds")
        logging.info(f"Final response: {json.dumps(response)}")
        
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': str(e), 'error': 'processing_error'}), 500

@app.route('/health', methods=['GET'])
def health():
    logging.info(f"Health check from: {request.remote_addr}")
    return jsonify({
        'success': True, 
        'message': 'AI service is running', 
        'model': MODEL_PATH,
        'status': 'online',
        'version': '1.0'
    })

@app.route('/test', methods=['GET'])
def test_model():
    logging.info("Test model endpoint called")
    try:
        # Create a simple test image (red square)
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([(50, 50), (174, 174)], fill='red')
        
        # Test prediction
        test_img = preprocess_image(img)
        preds = model.predict(test_img)
        pred_idx = np.argmax(preds[0])
        
        return jsonify({
            'success': True,
            'test_prediction': CLASS_NAMES[pred_idx],
            'model_path': MODEL_PATH,
            'message': 'Model loaded and working'
        })
    except Exception as e:
        logging.error(f"Test endpoint error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    logging.info(f"Starting Flask AI API with model: {MODEL_PATH}")
    logging.info(f"Classes: {CLASS_NAMES}")
    logging.info(f"API will be available at: http://0.0.0.0:5000/api/v1/ai-cake")
    logging.info(f"Health check endpoint: http://0.0.0.0:5000/health")
    logging.info(f"Test endpoint: http://0.0.0.0:5000/test")
    print("🚀 Starting AI Cake Matching Server...")
    print("📡 Server will be accessible on:")
    print("   - Main API: http://192.168.100.4:5000/api/v1/ai-cake")
    print("   - Health: http://192.168.100.4:5000/health")
    print("   - Test: http://192.168.100.4:5000/test")
    print("⚠️  Make sure Windows Firewall allows Python.exe")
    print("🔥 Server starting...")
    app.run(host='0.0.0.0', port=5000, debug=True)
