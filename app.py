from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import glob

app = Flask(__name__)

# Directory containing models
models_dir = r"D:/RGT/Code/Project/Cocoa_Bean_Prediction/models"
# Limit to specific models
model_files = ["cnn_model.h5", "mobilenet_model.h5"]

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
    x = x / 255.0  # Normalize to [0, 1]
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
            
            # Dictionary to store predictions from selected models
            predictions = {}
            
            # Load and predict using selected models
            for model_name in model_files:
                model_path = os.path.join(models_dir, model_name)
                model = tf.keras.models.load_model(model_path)
                class_prediction = model.predict(img)
                confidence = float(np.max(class_prediction)) * 100
                class_idx = np.argmax(class_prediction, axis=1)[0]

                # Store predictions with model name
                predictions[model_name] = {
                    "bean": class_idx,
                    "confidence": confidence
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

            return render_template('predict.html', bean=bean, confidence=confidence, best_model=best_model)
        else:
            return "File type not allowed or file could not be processed", 400

    return "Invalid request method", 405

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
