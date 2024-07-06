# Cocoa Bean Prediction

## Overview

Cocoa Bean Prediction is a web application that utilizes deep learning to classify cocoa beans into one of six categories. The application takes an image, either from a camera or a local directory, and processes it through a Flask API that hosts the predictive models.

## Categories

The cocoa beans can be classified into the following types:

0. Bean Fraction
1. Broken Bean
2. Fermented Bean
3. Moldy Bean
4. Unfermented Bean
5. Whole Bean

## Folder Structure
There are four folders in the directory. The `data  analysis and modelling` folder contains the notebook files for analysis, model definitions and training. The `models` folder contains the trained models. The `static` folder contains CSS files and images. The `templates` folder contains the html files for the home and prediction result pages.

<img src="https://github.com/Berchie-Sam/Cocoa_Bean_Prediction/blob/main/static/assets/folder_structure.jpg" width="400" heigth="600">


## Notebooks
### Data_pipeline.ipynb

This notebook focuses on the data pipeline required for processing and preparing the data for model training. It includes the following:

- **Data loading and cleaning**: Methods to load and clean the raw data.
- **Data preprocessing and augmentation**: Methods to prepare and augment the data for better model training.

### Cocoa_Beans_Classification.ipynb

This notebook contains the code for training the deep learning model used for classifying cocoa beans. It includes the following:

- **Model architecture definition**: Detailed architecture of the deep learning models used.
- Training and validation routines
- **Training and validation routines**: Procedures for training the models and validating their performance.
- **Evaluation metrics**: Metrics used to evaluate the performance of the models.


## app.py

The `app.py` file is the main Flask application that handles image uploads, predictions, and rendering the results. Key functionalities include:
- **File uploads**: Handling image file uploads and ensuring they are of the correct type (jpg, jpeg, png).
- **Image processing**: Loading and preparing the image for prediction.
- **Model loading and prediction**: Loading multiple models and predicting the class of the uploaded image.
- **Result rendering**: Rendering the prediction results, including the predicted class and confidence score.
## Templates

### index.html
The `index.html` file is the homepage of the application, where users can upload an image for classification. It includes:
- **Form for image upload**: Allows users to select and upload an image.
- **Image preview**: Displays a preview of the uploaded image.
- **Buttons**: Provides buttons to clear the preview or submit the form.
- **Loading indicator**: Displays a loading indicator while the prediction is processing.

<div style="text-align:center;">
    <img src="https://github.com/Berchie-Sam/Cocoa_Bean_Prediction/blob/main/static/assets/home.png" width="600" height="400">
</div>

### predict.html
The `predict.html` file displays the prediction results. It includes:
- **Predicted bean type**: Displays the predicted cocoa bean type.
- **Confidence score**: Shows the confidence score of the prediction.
- **Upload another image button**: Allows users to upload another image for classification.

<div style="text-align:center;">
    <img src="https://github.com/Berchie-Sam/Cocoa_Bean_Prediction/blob/main/static/assets/result_page.png" width="600" height="400">
</div>


## Static

### styles.css

The `styles.css` file contains custom styles for the web application.

### images

The `images` folder is used to store images that have been classified by the model. This will be replaced by a database in the future.

### assets

The `assets` folder is used to store images and other resources that maybe be important.

## Installation

### Prerequisites

- Python 3.7 or higher
- Flask
- TensorFlow
- OpenCV

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/cocoa-bean-prediction.git
    cd cocoa-bean-prediction
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Flask API:
    ```bash
    python app.py
    ```

## Usage
### Access the Web Application
Open your web browser and go to `http://127.0.0.1:8000`.

### Upload an Image
You can either take a picture using your camera or upload an image from your local directory.

### Get the Prediction
The application will classify the uploaded image into one of the six cocoa bean categories.

## Features
- **Image Upload**: Upload an image from your local device or use the camera.
- **Prediction**: Get real-time classification of cocoa beans.
- **API**: A Flask-based API to handle image processing and predictions.

## Contributing
We welcome contributions! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-branch
    ```
3. Commit your changes:
    ```bash
    git commit -am 'Add new feature'
    ```
4. Push to the branch:
    ```bash
    git push origin feature-branch
    ```
5. Create a new Pull Request.

- **Note**: Please ensure that your pull request targets the `main` branch of the repository when submitting contributions.


## Contact Information
For any inquiries or questions, please contact us at: [soberchie@gmail.com]