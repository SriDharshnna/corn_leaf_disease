from pathlib import Path
import io
import base64

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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    tf = None

model = None
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.65  # below this, the frontend shows the "ambiguous case" banner

if tf is not None:
    model_path = BASE_DIR / 'backend' / 'model' / 'corn_disease_model_FINAL.keras'
    print("Looking for model at:", model_path)
    print("File exists?", model_path.exists())
    if model_path.exists():
        try:
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print("MODEL LOAD FAILED:", e)
else:
    print("TensorFlow itself failed to import.")


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, np.array(img)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    base_model = model.get_layer('efficientnetb0')
    grad_model_base = tf.keras.models.Model(
        base_model.input, base_model.get_layer(last_conv_layer_name).output
    )
    preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_array)

    with tf.GradientTape() as tape:
        conv_outputs = grad_model_base(preprocessed)
        tape.watch(conv_outputs)
        x = model.get_layer('global_average_pooling2d')(conv_outputs)
        x = model.get_layer('dropout')(x, training=False)
        x = model.get_layer('dense')(x)
        x = model.get_layer('batch_normalization')(x, training=False)
        x = model.get_layer('dropout_1')(x, training=False)
        predictions = model.get_layer('dense_1')(x)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()[0]


def generate_gradcam_overlay(img_array, orig_img):
    try:
        heatmap = make_gradcam_heatmap(img_array, model)
        heatmap_resized = tf.image.resize(
            heatmap[..., tf.newaxis], (IMG_SIZE, IMG_SIZE)
        ).numpy().squeeze()

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(orig_img)
        ax.imshow(heatmap_resized, cmap='jet', alpha=0.5)
        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print("Grad-CAM generation failed:", e)
        return None


@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_array, orig_img = preprocess_image(file.read())

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
            'low_confidence': confidence < CONFIDENCE_THRESHOLD,
            'gradcam_image': None,
            'note': 'Using lightweight fallback classifier because the TensorFlow model is unavailable.'
        })

    preds = model.predict(img_array)[0]
    confidence = float(np.max(preds))
    predicted_class = class_names[int(np.argmax(preds))]

    gradcam_b64 = generate_gradcam_overlay(img_array, orig_img)

    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': {class_names[i]: float(preds[i]) for i in range(len(class_names))},
        'low_confidence': confidence < CONFIDENCE_THRESHOLD,
        'gradcam_image': gradcam_b64
    }
    return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')