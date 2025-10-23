from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import io
import urllib.request

# -----------------------------------------------------
# 1️⃣ Initialize Flask + Firebase
# -----------------------------------------------------
app = Flask(__name__)

SA_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
cred = credentials.Certificate(SA_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# -----------------------------------------------------
# 2️⃣ Load Models
# -----------------------------------------------------
print("Loading models...")

# --- TFLite Segmentation model ---
SEG_MODEL_PATH = "models/segmentation_model.tflite"
seg_interpreter = tf.lite.Interpreter(model_path=SEG_MODEL_PATH)
seg_interpreter.allocate_tensors()
input_details = seg_interpreter.get_input_details()
output_details = seg_interpreter.get_output_details()

# --- EfficientNetB0 Classifier ---
CLASSIFIER_PATH = "models/classification_model.h5"
classifier = load_model(CLASSIFIER_PATH)

print("✅ Models loaded successfully!")

# -----------------------------------------------------
# 3️⃣ Helper Functions
# -----------------------------------------------------
def preprocess_image(img, target_size=(224, 224)):
    """Resize and normalize image for model input"""
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

def run_segmentation(image_array):
    """Run TFLite DeepLabV3+ segmentation"""
    seg_interpreter.set_tensor(input_details[0]['index'], image_array)
    seg_interpreter.invoke()
    seg_output = seg_interpreter.get_tensor(output_details[0]['index'])
    return seg_output  # shape: [1, H, W, num_classes]

def run_classification(image_array):
    """Run EfficientNet classifier"""
    preds = classifier.predict(image_array)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    return class_idx, confidence

# Dummy class mapping (customize as per your dataset)
WOUND_CLASSES = ["Cut", "Burn", "Ulcer", "Infection"]

# -----------------------------------------------------
# 4️⃣ Main AI Endpoint
# -----------------------------------------------------
@app.route("/infer", methods=["POST"])
def infer():
    """
    Expected JSON:
    {
        "scanId": "scan001",
        "imageUrl": "https://.../some_image.jpg",
        "metadata": {...}  # optional additional info
    }
    """
    data = request.get_json()
    scan_id = data.get("scanId")
    image_url = data.get("imageUrl")
    metadata = data.get("metadata", {})

    if not scan_id or not image_url:
        return jsonify({"error": "scanId and imageUrl required"}), 400

    # --- Download image ---
    try:
        image_bytes = urllib.request.urlopen(image_url).read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to download or open image: {str(e)}"}), 400

    # --- Preprocess ---
    img_array = preprocess_image(image)

    # --- Segmentation ---
    seg_output = run_segmentation(img_array)

    # --- Classification ---
    class_idx, confidence = run_classification(img_array)
    wound_type = WOUND_CLASSES[class_idx] if class_idx < len(WOUND_CLASSES) else "Unknown"

    # --- Simple Prognosis ---
    # You can enhance this with metadata or time-series later
    prognosis = "Healing" if confidence > 0.8 else "Needs Attention"
    severity = "Moderate"  # Placeholder, can be computed from segmentation mask

    # --- Final AI Results ---
    ai_results = {
        "woundType": wound_type,
        "severity": severity,
        "prognosis": prognosis,
        "confidence": confidence,
    }

    # --- Update Firestore ---
    try:
        scan_ref = db.collection("scans").document(scan_id)
        scan_ref.set({
            "scanId": scan_id,
            "imageUrl": image_url,
            "metadata": metadata,
            "aiResults": ai_results,
            "status": "completed",
            "updatedAt": firestore.SERVER_TIMESTAMP
        }, merge=True)
    except Exception as e:
        return jsonify({"error": f"Firestore update failed: {str(e)}"}), 500

    return jsonify({
        "status": "done",
        "aiResults": ai_results
    })

# -----------------------------------------------------
# 5️⃣ Run Flask
# -----------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port)
