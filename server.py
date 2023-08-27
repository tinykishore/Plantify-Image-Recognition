from flask import Flask, request, jsonify
import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image
import io
import base64
from flask_cors import CORS  # Import the CORS class

app = Flask(__name__)
CORS(app)

# Load the model from TensorFlow Hub
model = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1')


@app.route('/predict', methods=['POST'])
def predict():
    print("Request received")
    data = request.json
    base64_image = data['image']

    # strip the header
    base64_image = base64_image.split(",")[1]

    # Decode base64 image string
    image_bytes = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    image = np.array(image) / 255.0
    image = image[np.newaxis, ...]

    # Make predictions
    predictions = model(image)
    predicted_class_index = int(np.argmax(predictions, axis=1)[0])  # Convert to regular Python int

    # Load the class labels mapping

    class_labels_path = "labels.txt"
    class_labels = {}
    with open(class_labels_path) as f:
        next(f)  # Skip the header line
        for line in f:
            index, name = line.strip().split(',')
            class_labels[int(index.strip())] = name.strip()

    # Get the plant name using the predicted class index
    predicted_plant_name = class_labels[predicted_class_index]

    response = {
        "predicted_class_index": predicted_class_index,
        "predicted_plant_name": predicted_plant_name
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
