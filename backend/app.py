from pathlib import Path
import io

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / 'frontend'

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='/')
CORS(app)

try:
    import tensorflow as tf
except Exception:
    tf = None

model = None
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
IMG_SIZE = 224

if tf is not None:
    model_path = BASE_DIR / 'backend' / 'model' / 'corn_disease_model_FINAL.keras'
    if model_path.exists():
        model = tf.keras.models.load_model(model_path)


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_array = preprocess_image(file.read())

    if model is None:
        gray = float(np.mean(img_array[0]))
        green_score = float(np.mean(img_array[0, :, :, 1]))
        if green_score > 135:
            predicted_class = 'Healthy'
            confidence = 0.72
        elif gray > 120:
            predicted_class = 'Blight'
            confidence = 0.6
        else:
            predicted_class = 'Common_Rust'
            confidence = 0.58

        probabilities = {
            'Blight': 0.2,
            'Common_Rust': 0.2,
            'Gray_Leaf_Spot': 0.2,
            'Healthy': 0.4,
        }
        probabilities[predicted_class] = confidence
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': probabilities,
            'note': 'Using lightweight fallback classifier because the TensorFlow model is unavailable.'
        })

    preds = model.predict(img_array)[0]
    result = {
        'predicted_class': class_names[np.argmax(preds)],
        'confidence': float(np.max(preds)),
        'all_probabilities': {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    }
    return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')