import os
# Suppress oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

save_path = f"cnn_1_model.keras"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 1

# Path to the image for prediction
path = ["test_images/big-live-alive-raw-fresh-260nw-789090217.png"]
label = ['dummy' for _ in range(len(path))]

# Prepare dataframe for the test image
x_test = pd.DataFrame({'path': path, 'label': label})

# Create the ImageDataGenerator for preprocessing (rescaling)
test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.0)

# Generate the test data (images only, no labels)
test = test_data_gen.flow_from_dataframe(
    dataframe=x_test,
    x_col='path',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    class_mode='categorical', 
    batch_size=BATCH_SIZE,
    shuffle=False  # Ensure no shuffling
)

# Load the model
model = load_model(save_path)

# Perform prediction (for all images in the generator)
pred = model.predict(test, steps=len(test))  # Predict for all images
pred = np.argmax(pred, axis=1)  # Get the predicted class index

# Define labels and reverse label encoding
labels = ['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel',
          'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp',
          'Striped Red Mullet', 'Trout']

unique_labels = sorted(set(labels))
label_encoder = {label: idx for idx, label in enumerate(unique_labels)}

reverse_label_encoder = {idx: label for label, idx in label_encoder.items()}

# Convert the prediction index to the actual class name
pred_class_name = reverse_label_encoder[pred[0]]
print(f"Predicted Class: {pred_class_name}")