from PIL import Image
from io import BytesIO
import base64
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import logging
from datetime import datetime

app = Flask(__name__)

# Directory containing models (relative path)
models_dir = os.path.join(os.getcwd(), "models")
is_bean_model_dir = os.path.join(os.getcwd(), "static/check_bean")

# Specific model files to use
models_path = [
    os.path.join(models_dir, "cnn_model.h5"),
    os.path.join(models_dir, "mobilenet_model.h5")
]

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

# Directory to save captured images
captured_img_dir = os.path.join(target_img, 'D:\RGT\Code\Project\Cocoa_Bean_Prediction\static\images\captured')
if not os.path.exists(captured_img_dir):
    os.makedirs(captured_img_dir)

@app.route('/save_captured_image', methods=['POST'])
def save_captured_image():
    try:
        image_data = request.json['imageData']
        if image_data.startswith('data:image/'):
            # Remove the data URL prefix
            image_data = image_data.split(',')[1]
            # Decode the base64 string
            image_bytes = base64.b64decode(image_data)
            # Generate a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_{timestamp}.png"
            file_path = os.path.join(captured_img_dir, filename)
            # Save the image
            with open(file_path, 'wb') as f:
                f.write(image_bytes)
            logging.info(f"Captured image saved: {file_path}")
            return jsonify({"success": True, "filename": filename})
        else:
            logging.warning("Invalid image data format")
            return jsonify({"success": False, "error": "Invalid image data format"}), 400
    except Exception as e:
        logging.error(f"Error saving captured image: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("Received prediction request")

        file_path = None

        if 'file' in request.files:
            logging.info("File upload detected")
            file = request.files['file']
            if file.filename != '':
                if allowed_file(file.filename):
                    filename = file.filename
                    file_path = os.path.join(target_img, filename)
                    file.save(file_path)
                    logging.info(f"File saved: {file_path}")
                else:
                    logging.error(f"Invalid file type: {file.filename}")
                    return jsonify({"success": False, "error": "Invalid file type. Allowed types are jpg, jpeg, png"}), 400

        elif 'filename' in request.form:
            # Use the saved captured image
            filename = request.form['filename']
            file_path = os.path.join(captured_img_dir, filename)
            logging.info(f"Using captured image: {file_path}")

        else:
            logging.warning("No file uploaded or captured image provided")
            return jsonify({"success": False, "error": "No file uploaded or captured image provided"}), 400

        if not file_path or not os.path.exists(file_path):
            logging.error("Invalid file path")
            return jsonify({"success": False, "error": "Invalid file path"}), 400

        # Process the image for prediction
        img = read_image(file_path)
        
        logging.info("Loading is_bean_model")
        is_bean_model = tf.keras.models.load_model(os.path.join(is_bean_model_dir, 'is_bean_model.h5'))
        is_bean_prediction = is_bean_model.predict(img)
        is_bean_class = np.argmax(is_bean_prediction, axis=1)[0]  # 0 for Cocoa Bean, 1 for Non-Cocoa Bean

        if is_bean_class == 1:  # Non-Cocoa Bean
            logging.info("Non-Cocoa Bean detected")
            return jsonify({"success": False, "message": "The image is not a Cocoa Bean, please upload a valid image."}), 400

        else:
            # If it's a Cocoa Bean, proceed with other models
            logging.info("Cocoa Bean detected, proceeding to classify bean type")

            # Dictionary to store predictions from selected models
            predictions = {}
            
            # Load and predict using selected models
            for model_path in models_path:
                logging.info(f"Loading model from: {model_path}")
                model_name = os.path.basename(model_path).split('.')[0]
                logging.info(f"Model name: {model_name}")
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
            if predictions:
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
                return jsonify({"success": False, "error": "No valid models found for prediction"}), 500

    except Exception as e:
        logging.info(f"Error during prediction: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)