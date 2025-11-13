# app.py
import os
import time
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
import pandas as pd  
import numpy as np
from utils import create_data_batches
from PIL import Image

# -----------------------------
# CONFIGURATION
# -----------------------------
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/17082025-220649-full-image-set-mobilenet-Adam.h5'
EXCEL_PATH = 'model/labels.csv'   
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

labels_df = pd.read_csv(EXCEL_PATH)
# breed_names = labels_df['breed'].unique()
# num_classes = len(breed_names)

# -----------------------------
# LOAD MODEL AND CLASS NAMES
# -----------------------------
print("Loading model...")
# Load the full trained model (SavedModel format is recommended)
model = keras.models.load_model(
    'model/dog_breed_model.h5',  # path to SavedModel folder
    custom_objects={'KerasLayer': hub.KerasLayer}
)
# model.load_weights('model/dog_breed_model.h5')

print("Loading breed names from Excel...")
df = pd.read_csv(EXCEL_PATH)


CLASS_NAMES = np.unique(labels_df['breed']).tolist() 



print(f"Loaded {len(CLASS_NAMES)} breed names.")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def get_pred_label(prediction_probabilities):
    return CLASS_NAMES[np.argmax(prediction_probabilities)]

def is_valid_image(filepath):
    try:
        img = Image.open(filepath)
        return img.format in ["JPEG", "JPG", "PNG"]   # True only if ACTUAL JPEG
    except:
        return False

# -----------------------------
# ROUTES
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to avoid overwriting files
        filename = f"{int(time.time())}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # 🚨 Check if the uploaded file is a REAL JPEG (not fake renamed)
        if not is_valid_image(filepath):
            os.remove(filepath)  # delete fake file
            return render_template("index.html", error="❌  Only real JPG, JPEG, or PNG images are allowed!")

        # ✅ Create test batch using your existing batching function
        data_batch = create_data_batches([filepath], test_data=True)

        # ✅ Get prediction probabilities
        predictions = model.predict(data_batch)
        preds = np.squeeze(predictions)  # shape -> (num_classes,)

        # ✅ Use your get_pred_label() function for the top prediction
        top_label = get_pred_label(preds)

        # ✅ Get top 3 predictions for display
        top_k = preds.argsort()[-3:][::-1]
        top_results = []
        for idx in top_k:
            label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"label_{idx}"
            conf = float(preds[idx])
            top_results.append({
                'label': label,
                'confidence': round(conf * 100, 2)
            })

        # ✅ Generate image URL for template display
        image_url = url_for('static', filename=f'uploads/{filename}')

        return render_template(
            'result.html',
            image_url=image_url,
            predicted_label=top_label,
            results=top_results
        )

    return redirect(url_for('index'))


# -----------------------------
# MAIN
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
