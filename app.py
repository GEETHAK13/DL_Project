from flask import Flask, request, jsonify, render_template
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
import joblib
app = Flask(__name__)


# Path to your v1.pkl file
file_path = 'v1.pkl'

# Load the model
model = joblib.load(file_path)


# Define labels
labels = ['organic', 'recyclable']

def preprocess_image(image_data):
    # Convert base64 string to image
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    # Resize image to match model input shape
    image = image.resize((224, 224))
    # Convert image to numpy array
    image_array = np.asarray(image) / 255.0  # Normalize pixel values
    # Expand dimensions to match model input shape [batch_size, height, width, channels]
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def index():
    return render_template('main.html')

# @app.route('/index')
# def index1():
#     return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("HI")
    # Extract image data from request
    image_data = request.files['image'].read()
    # Preprocess image
    processed_image = preprocess_image(base64.b64encode(image_data).decode('utf-8'))
    # Make prediction
    prediction = model.predict(processed_image)
    # Get label with highest probability
    predicted_label = labels[np.argmax(prediction)]
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    return render_template('result.html', prediction=predicted_label, encoded_image=encoded_image)

if __name__ == '__main__':
    app.run(debug=True)