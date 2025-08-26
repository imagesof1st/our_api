import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ------------------------------
# CONFIG
# ------------------------------
MODEL_PATH = "models/mobilenet_model.h5"   # path to saved model
IMG_SIZE = (224, 224)

# Class labels
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
# INIT FLASK
# ------------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Load model once when API starts
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ------------------------------
# ROUTES
# ------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Waste Classification API is running 🚀"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    
    file = request.files["file"]

    try:
        # Load and preprocess image
        img = image.load_img(file, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        preds = model.predict(img_array)
        predicted_class = np.argmax(preds, axis=1)[0]
        confidence = float(preds[0][predicted_class])

        return jsonify({
            "success": True,
            "label": class_labels[predicted_class],
            "confidence": round(confidence * 100, 2)
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------------------
# RUN (for local dev)
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns PORT dynamically
    app.run(host="0.0.0.0", port=port)
