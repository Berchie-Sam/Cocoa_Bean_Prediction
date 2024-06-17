# app.py

from flask import Flask, render_template, request, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import os
import glob
import base64

app = Flask(__name__)

# Directory containing models
models_dir = r"D:/RGT/Code/Project/Cocoa_Bean_Prediction/models"
models_path = os.path.join(models_dir, "*.h5")

# Target image size
image_size = (114, 114)

# Directory to save uploaded images
target_img = os.path.join(os.getcwd(), 'static/images')

# Ensure the directory exists
if not os.path.exists(target_img):
    os.makedirs(target_img)

# Allow files with extension png, jpg, and jpeg
ALLOWED_EXT = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# Function to load and prepare the image in the right shape
def read_image(filename):
    img = load_img(filename, target_size=image_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']
        
        if file.filename == '':
            return "No selected file", 400
        
        if file and allowed_file(file.filename):
            # Read image file and save to target_img directory
            filename = file.filename
            file_path = os.path.join(target_img, filename)
            file.save(file_path)

            # Read the image again to prepare for prediction
            img = read_image(file_path)
            
            # Encode image as base64 to pass to predict.html
            with open(file_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

            # Dictionary to store predictions from all models
            predictions = {}
            
            # Load and predict using all models
            for model_path in glob.glob(models_path):
                model = tf.keras.models.load_model(model_path)
                class_prediction = model.predict(img)
                confidence = float(np.max(class_prediction)) * 100
                class_idx = np.argmax(class_prediction, axis=1)[0]

                # Store predictions with model name
                predictions[os.path.basename(model_path)] = {
                    "bean": class_idx,
                    "confidence": confidence,
                    'user_image': encoded_image  # Pass base64-encoded image data
                }
            
            # Determine the best prediction
            best_model, best_pred = max(predictions.items(), key=lambda item: item[1]['confidence'])
            best_bean_class = best_pred['bean']
            confidence = best_pred['confidence']
            
            bean_dict = {
                0: "Bean Fraction",
                1: "Broken Bean",
                2: "Fermented Bean",
                3: "Moldy Bean",
                4: "Unfermented Bean",
                5: "Whole Bean"
            }
            
            bean = bean_dict.get(best_bean_class, "Error")

            return render_template('predict.html', bean=bean, confidence=confidence, user_image=encoded_image, best_model=best_model)
        else:
            return "File type not allowed or file could not be processed", 400

    return "Invalid request method", 405

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
