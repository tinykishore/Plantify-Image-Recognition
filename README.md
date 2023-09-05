# Plant Image Recognition for Plantify
###### Model for plant recognition in Plantify project with pre-built model provided by Google Tensorflow

[View in TensorFlow Hub website](https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1)

## Overview
This model is trained to recognize 2101 plants species from images. It is based on MobileNet, and trained on photos contributed by the iNaturalist community.

The species and images are a subset of the iNaturalist 2017 Competition dataset, organized by Visipedia. This model was originally published as part of the Natural Explorer module for Google's AIY Vision Kit.

## Input
This model takes input of images. Inputs are expected to be 3-channel RGB color images of size 224 x 224, scaled to [0, 1].

## Output
This model outputs to `image_classifier`.

`image_classifier`: A probability vector of dimension 2102, corresponding to a background class and 2101 plant species in the labelmap. See `labels.txt` for the list of plant species.

## Mechanism
- It imports the necessary libraries: `tensorflow`, `tensorflow_hub` as `hub`, `numpy` as `np`, and `PIL.Image` as `Image`
- It loads the pre-trained plant classification model from **TensorFlow Hub** using the `hub.KerasLayer()` function. The model used in this example is the "_plants_V1_" model from the **AIY Vision Kit**
- It defines the path to the input image and opens it using the `PIL.Image.open()` function. The image is then resized to (224, 224) pixels
- The pixel values of the image are normalized between 0 and 1 by dividing the image array by 255.0
- The image array is expanded to include a batch dimension by adding an extra dimension using `np.newaxis`
- The pre-trained model is used to make predictions on the input image by passing the image array to the model object.
- The predicted class index is obtained by finding the index with the highest predicted probability using `np.argmax(predictions, axis=1)[0]`
- The code then loads a class labels mapping from a file. Each line in the file represents a class label, and the index and name are separated by a comma. The class labels are stored in a dictionary with the index as the key and the name as the value
- The predicted class index is used to retrieve the corresponding plant name from the class labels' dictionary
- Finally, the predicted plant name is printed

> THIS MODEL CANNOT BE RE-TRAINED
