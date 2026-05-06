"""
LeafLens — Flask REST API (Backend)
Run: python app.py
Endpoint: POST /predict  — accepts base64 image, returns JSON diagnosis
"""

import os
import io
import json
import base64
import logging
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# APP INIT
# ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow Streamlit frontend to call this API

# ──────────────────────────────────────────────
# LOAD MODEL + CLASS NAMES
# ──────────────────────────────────────────────
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "../model/leaflens_model.h5")
CLASS_JSON_PATH = os.path.join(os.path.dirname(__file__), "../model/class_names.json")

logger.info("Loading model…")
model = tf.keras.models.load_model(MODEL_PATH)
logger.info("Model loaded.")

with open(CLASS_JSON_PATH) as f:
    CLASS_NAMES = json.load(f)   # {"0": "Apple___Apple_scab", ...}

# ──────────────────────────────────────────────
# TREATMENT DATABASE
# One entry per class.  Key = class name from PlantVillage.
# ──────────────────────────────────────────────
TREATMENTS = {
    "Apple___Apple_scab": {
        "severity": "Moderate",
        "treatment": "Apply fungicides containing captan or myclobutanil. Remove and destroy infected leaves. Ensure proper tree spacing for air circulation.",
        "prevention": "Plant resistant varieties. Apply preventive fungicide sprays in spring."
    },
    "Apple___Black_rot": {
        "severity": "High",
        "treatment": "Prune out infected branches 8–12 inches below visible symptoms. Apply captan-based fungicide. Remove mummified fruit.",
        "prevention": "Maintain good sanitation. Avoid tree injuries."
    },
    "Apple___Cedar_apple_rust": {
        "severity": "Moderate",
        "treatment": "Apply fungicides (myclobutanil, triadimefon) at pink stage. Remove nearby juniper/cedar trees if possible.",
        "prevention": "Plant rust-resistant apple varieties."
    },
    "Apple___healthy": {
        "severity": "None",
        "treatment": "No treatment needed. Your plant looks healthy!",
        "prevention": "Continue good cultural practices."
    },
    "Blueberry___healthy": {
        "severity": "None",
        "treatment": "Plant is healthy. No action required.",
        "prevention": "Maintain proper soil pH (4.5–5.5) and irrigation."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "severity": "Moderate",
        "treatment": "Apply sulfur or potassium bicarbonate fungicides. Improve air circulation by pruning.",
        "prevention": "Avoid overhead irrigation. Plant in sunny locations."
    },
    "Cherry_(including_sour)___healthy": {
        "severity": "None",
        "treatment": "Plant is healthy!",
        "prevention": "Monitor regularly for pests and diseases."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "severity": "High",
        "treatment": "Apply strobilurin or triazole fungicides. Rotate crops. Plow under infected debris.",
        "prevention": "Use resistant hybrids. Maintain good field drainage."
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "Moderate",
        "treatment": "Apply fungicides (mancozeb, chlorothalonil) early. Plant early-maturing varieties.",
        "prevention": "Use resistant varieties. Scout fields regularly."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "severity": "High",
        "treatment": "Apply fungicides at tasseling stage. Remove crop debris after harvest.",
        "prevention": "Rotate corn with non-host crops. Use tolerant hybrids."
    },
    "Corn_(maize)___healthy": {
        "severity": "None",
        "treatment": "Crop looks healthy!",
        "prevention": "Maintain balanced fertilization and proper irrigation."
    },
    "Grape___Black_rot": {
        "severity": "High",
        "treatment": "Apply captan, mancozeb, or myclobutanil. Remove mummified berries and infected canes.",
        "prevention": "Prune for good air circulation. Apply fungicides from bud break."
    },
    "Grape___Esca_(Black_Measles)": {
        "severity": "High",
        "treatment": "No cure exists. Remove and destroy infected vines. Apply wound protectants after pruning.",
        "prevention": "Prune during dry weather. Avoid large pruning wounds."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "severity": "Moderate",
        "treatment": "Apply copper-based fungicides. Remove infected leaves.",
        "prevention": "Ensure proper vine spacing. Avoid leaf wetness."
    },
    "Grape___healthy": {
        "severity": "None",
        "treatment": "Vine is healthy!",
        "prevention": "Regular monitoring and balanced nutrition."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "severity": "Critical",
        "treatment": "No cure. Remove and destroy infected trees immediately to prevent spread. Control Asian citrus psyllid vector with insecticides.",
        "prevention": "Use disease-free nursery stock. Apply systemic insecticides for psyllid control."
    },
    "Peach___Bacterial_spot": {
        "severity": "High",
        "treatment": "Apply copper-based bactericides. Avoid overhead irrigation.",
        "prevention": "Plant resistant varieties. Prune for air circulation."
    },
    "Peach___healthy": {
        "severity": "None",
        "treatment": "Tree is healthy!",
        "prevention": "Regular pruning and disease scouting."
    },
    "Pepper,_bell___Bacterial_spot": {
        "severity": "High",
        "treatment": "Apply copper hydroxide sprays. Remove infected plant material. Rotate crops.",
        "prevention": "Use certified disease-free seeds. Avoid working in wet fields."
    },
    "Pepper,_bell___healthy": {
        "severity": "None",
        "treatment": "Plant is healthy!",
        "prevention": "Maintain good soil drainage and nutrition."
    },
    "Potato___Early_blight": {
        "severity": "Moderate",
        "treatment": "Apply chlorothalonil or mancozeb fungicides. Remove infected lower leaves.",
        "prevention": "Use certified seed potatoes. Maintain adequate plant nutrition."
    },
    "Potato___Late_blight": {
        "severity": "Critical",
        "treatment": "Apply metalaxyl or cymoxanil immediately. Destroy infected plants. Avoid overhead irrigation.",
        "prevention": "Plant resistant varieties. Scout fields regularly, especially in wet weather."
    },
    "Potato___healthy": {
        "severity": "None",
        "treatment": "Crop is healthy!",
        "prevention": "Use certified seed potatoes and practice crop rotation."
    },
    "Raspberry___healthy": {
        "severity": "None",
        "treatment": "Plant is healthy!",
        "prevention": "Maintain proper pruning and irrigation."
    },
    "Soybean___healthy": {
        "severity": "None",
        "treatment": "Crop is healthy!",
        "prevention": "Rotate with non-legume crops. Monitor for pests."
    },
    "Squash___Powdery_mildew": {
        "severity": "Moderate",
        "treatment": "Apply sulfur or neem oil sprays. Remove heavily infected leaves.",
        "prevention": "Plant resistant varieties. Avoid overhead watering."
    },
    "Strawberry___Leaf_scorch": {
        "severity": "Moderate",
        "treatment": "Apply captan or thiram fungicides. Remove infected leaves.",
        "prevention": "Plant in well-drained soil. Avoid leaf wetness."
    },
    "Strawberry___healthy": {
        "severity": "None",
        "treatment": "Plant is healthy!",
        "prevention": "Maintain proper spacing and irrigation."
    },
    "Tomato___Bacterial_spot": {
        "severity": "High",
        "treatment": "Apply copper bactericides + mancozeb. Remove infected plant parts. Avoid working when wet.",
        "prevention": "Use resistant varieties and disease-free seed."
    },
    "Tomato___Early_blight": {
        "severity": "Moderate",
        "treatment": "Apply chlorothalonil or copper-based fungicides. Remove lower infected leaves. Mulch soil.",
        "prevention": "Rotate crops. Use resistant varieties."
    },
    "Tomato___Late_blight": {
        "severity": "Critical",
        "treatment": "Apply metalaxyl or cymoxanil immediately. Remove all infected plant material. Destroy do not compost.",
        "prevention": "Plant resistant varieties. Avoid overhead irrigation."
    },
    "Tomato___Leaf_Mold": {
        "severity": "Moderate",
        "treatment": "Improve greenhouse ventilation. Apply fungicides (chlorothalonil, mancozeb).",
        "prevention": "Reduce humidity. Space plants properly."
    },
    "Tomato___Septoria_leaf_spot": {
        "severity": "High",
        "treatment": "Apply chlorothalonil or copper fungicides. Remove infected leaves. Avoid splashing water.",
        "prevention": "Rotate crops 2–3 years. Stake plants for air circulation."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "severity": "Moderate",
        "treatment": "Apply miticides (abamectin, bifenazate). Use insecticidal soap or neem oil for organic control.",
        "prevention": "Avoid water stress. Introduce predatory mites."
    },
    "Tomato___Target_Spot": {
        "severity": "High",
        "treatment": "Apply fungicides (azoxystrobin, chlorothalonil). Remove and destroy infected debris.",
        "prevention": "Rotate crops. Avoid overhead irrigation."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "Critical",
        "treatment": "No cure. Remove infected plants immediately. Control whitefly vectors with systemic insecticides.",
        "prevention": "Use virus-resistant varieties. Install insect-proof nets."
    },
    "Tomato___Tomato_mosaic_virus": {
        "severity": "High",
        "treatment": "No chemical cure. Remove infected plants. Disinfect tools with 10% bleach solution.",
        "prevention": "Use virus-free seed. Wash hands before handling plants."
    },
    "Tomato___healthy": {
        "severity": "None",
        "treatment": "Plant is healthy!",
        "prevention": "Continue good cultural practices and regular monitoring."
    }
}

SEVERITY_COLORS = {
    "None":     "green",
    "Moderate": "orange",
    "High":     "red",
    "Critical": "darkred"
}

CONFIDENCE_THRESHOLD = 0.50   # below this = uncertain prediction

# ──────────────────────────────────────────────
# IMAGE PREPROCESSING
# ──────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw bytes → normalized numpy tensor (1, 224, 224, 3)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # add batch dimension

# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "leaflens_model.h5"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 → bytes
        image_bytes = base64.b64decode(data["image"])

        # Preprocess
        tensor = preprocess_image(image_bytes)

        # Inference
        predictions = model.predict(tensor, verbose=0)[0]   # shape: (38,)

        top_idx        = int(np.argmax(predictions))
        confidence     = float(predictions[top_idx])
        class_name     = CLASS_NAMES.get(str(top_idx), "Unknown")

        # Top-3 predictions
        top3_idx = np.argsort(predictions)[::-1][:3]
        top3 = [
            {
                "class": CLASS_NAMES.get(str(i), "Unknown"),
                "confidence": round(float(predictions[i]) * 100, 2)
            }
            for i in top3_idx
        ]

        # Low confidence guard
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                "status":     "uncertain",
                "message":    "Confidence too low. Please upload a clearer, closer image of the leaf.",
                "confidence": round(confidence * 100, 2),
                "top3":       top3
            })

        # Treatment info
        info = TREATMENTS.get(class_name, {
            "severity":   "Unknown",
            "treatment":  "Consult a local agronomist for advice.",
            "prevention": "General good agricultural practices apply."
        })

        # Format class name for display
        display_name = class_name.replace("___", " — ").replace("_", " ")

        logger.info(f"Predicted: {class_name} ({confidence*100:.1f}%)")

        return jsonify({
            "status":       "success",
            "disease":      class_name,
            "display_name": display_name,
            "confidence":   round(confidence * 100, 2),
            "severity":     info["severity"],
            "treatment":    info["treatment"],
            "prevention":   info["prevention"],
            "top3":         top3
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/classes", methods=["GET"])
def get_classes():
    """Return all 38 class names — useful for debugging."""
    return jsonify(CLASS_NAMES)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)