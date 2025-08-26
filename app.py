from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import io
import os

# ------------------------------
# Initialize Flask app
# ------------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ------------------------------
# Load Waste Classification Model
# ------------------------------
MODEL_PATH = "models/check_model.h5"   # <-- path to your saved model 
model = load_model(MODEL_PATH)

# ------------------------------
# Waste class labels
# ------------------------------
class_labels = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass"
]

# ------------------------------
# Preprocess input image
# ------------------------------
def preprocess_image(image, target_size=(224, 224)):
    """Resize, convert to array, normalize and expand dims"""
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ------------------------------
# Prediction Route
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    try:
        # Load image from file bytes
        image_bytes = file.read()
        img = load_img(io.BytesIO(image_bytes), target_size=(224, 224))
        processed_image = preprocess_image(img)

        # Predict
        prediction = model.predict(processed_image)[0]
        predicted_index = np.argmax(prediction)
        confidence = float(prediction[predicted_index])

        return jsonify({
            'success': True,
            'label': class_labels[predicted_index],
            'confidence': round(confidence * 100, 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ------------------------------
# Run the app
# ------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
