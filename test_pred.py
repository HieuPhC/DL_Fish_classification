import os
# Suppress oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

save_path = f"cnn_1_model.keras"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 1

def preprocess_image_with_padding(image_path, target_height=IMG_HEIGHT, target_width=IMG_WIDTH):
    # Load the image
    img = tf.keras.preprocessing.image.load_img(image_path)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Add padding and resize
    padded_img = tf.image.resize_with_pad(img_array, target_height, target_width)
    
    # Normalize pixel values to [0, 1]
    padded_img = padded_img / 255.0
    
    return padded_img

# Path to the test image
image_path = "NA_Fish_Dataset/Red Sea Bream/100_1476.JPG"

# Preprocess the image
processed_image = preprocess_image_with_padding(image_path)

# Expand dimensions to simulate batch size of 1
processed_image = np.expand_dims(processed_image, axis=0)

# Load the model
model = load_model(save_path)

pred = model.predict(processed_image)
pred = np.argmax(pred, axis=1)

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