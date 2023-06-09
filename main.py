import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image

# Load the model from TensorFlow Hub
model = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1')

# Load and preprocess the image
image_path = "image.jpg"
image = Image.open(image_path).resize((224, 224))
image = np.array(image) / 255.0  # Normalize pixel values between 0 and 1
image = image[np.newaxis, ...]  # Add batch dimension

# Make predictions
predictions = model(image)

# Get the predicted class index
predicted_class_index = np.argmax(predictions, axis=1)[0]  # Access the first element of the array

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

print("Predicted plant name:", predicted_plant_name)
