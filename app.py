"""
Lung Cancer Prediction - Flask Web Application
Deep Learning + Groq LLM Clinical Explanation
"""

import os
from dotenv import load_dotenv

# ==========================================================
# FORCE LOAD .env FROM CURRENT DIRECTORY
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ==========================================================
# GROQ SETUP
# ==========================================================
client = None

try:
    from openai import OpenAI

    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key and groq_key.strip() != "":
        print("🔎 GROQ KEY FOUND (masked):", groq_key[:6] + "********")

        client = OpenAI(
            api_key=groq_key.strip(),
            base_url="https://api.groq.com/openai/v1"
        )

        print("✅ Groq client initialized successfully")
    else:
        print("❌ GROQ_API_KEY missing or empty in .env")

except Exception as e:
    print("❌ Groq setup failed:", str(e))
    client = None

# ==========================================================
# FLASK & ML IMPORTS
# ==========================================================
from flask import Flask, render_template, request, jsonify
import logging
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
IMAGE_SIZE = (350, 350)
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.hdf5')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CLASS_LABELS = {
    0: 'Adenocarcinoma',
    1: 'Large Cell Carcinoma',
    2: 'Normal',
    3: 'Squamous Cell Carcinoma'
}

# ==========================================================
# LLM FUNCTION
# ==========================================================
def generate_llm_report(disease, confidence):

    if client is None:
        return ("AI explanation unavailable (Groq key missing).", "no_api_key")

    try:
        if disease == "Normal":
            user_prompt = f"""
Chest X-ray result: NORMAL
Confidence: {confidence}%

Provide:
- 3 lung health maintenance tips
- When routine screening is recommended
"""
        else:
            user_prompt = f"""
Chest X-ray model prediction: {disease}
Confidence: {confidence}%

Provide:
- Possible symptoms
- Common causes
- When to consult a doctor
- Basic treatment overview
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a professional pulmonologist writing a clinical summary."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )

        llm_text = response.choices[0].message.content
        return (llm_text, None)

    except Exception as e:
        print("🔥 GROQ LLM ERROR:", str(e))
        return (f"LLM Error: {str(e)}", "service_error")

# ==========================================================
# MODEL LOADING
# ==========================================================
model = None

def load_model():
    global model

    if model is not None:
        return model

    try:
        print("🔄 Loading model...")

        base = tf.keras.applications.Xception(
            weights='imagenet',
            include_top=False,
            input_shape=[*IMAGE_SIZE, 3]
        )
        base.trainable = False

        model = Sequential([
            base,
            GlobalAveragePooling2D(),
            Dense(4, activation='softmax')
        ])

        if os.path.exists(MODEL_PATH):
            model.load_weights(MODEL_PATH)
            print("✅ Model loaded successfully")
        else:
            print("❌ Model file not found:", MODEL_PATH)

        return model

    except Exception as e:
        print("❌ Model load error:", str(e))
        return None


# ==========================================================
# UTIL FUNCTIONS
# ==========================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr


def get_image_base64(img_path):
    with Image.open(img_path) as img:
        img.thumbnail((400, 400))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ==========================================================
# ROUTES
# ==========================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"success": False, "error": "Model not loaded"}), 500

        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        arr = preprocess_image(path)
        preds = model.predict(arr, verbose=0)

        idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][idx] * 100)

        llm_text, llm_error = generate_llm_report(
            CLASS_LABELS[idx],
            round(confidence, 2)
        )

        result = {
            "success": True,
            "predicted_class": CLASS_LABELS[idx],
            "class_index": idx,
            "confidence": round(confidence, 2),
            "llm_report": llm_text,
            "llm_error": llm_error,
            "all_predictions": {
                CLASS_LABELS[i]: round(float(preds[0][i] * 100), 2)
                for i in range(4)
            },
            "image": get_image_base64(path)
        }

        os.remove(path)
        return jsonify(result)

    except Exception as e:
        print("❌ Prediction Error:", str(e))
        return jsonify({
            "success": False,
            "error": str(e),
            "predicted_class": None,
            "confidence": 0,
            "llm_report": "Unavailable"
        }), 500


@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


# ==========================================================
# PRELOAD MODEL
# ==========================================================
try:
    load_model()
except Exception as e:
    print("❌ Model preload failed:", str(e))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)