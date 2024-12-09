import os
# Suppress oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras.models import load_model

save_path = f"cnn-1(2).keras"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
path = ["Fish_Dataset/Shrimp/Shrimp/00001.png"]
x_test = pd.DataFrame({'path': path})


def preprocess_image(file_path):
    # Read and decode the image
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0  # Rescale the image pixels to [0, 1]
    return img

def create_dataset(dataframe, batch_size=BATCH_SIZE):
    paths = dataframe['path'].values
    
    # Create TensorFlow dataset for paths
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda x: preprocess_image(x), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# Load the saved model
labels = ['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel',
       'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp',
       'Striped Red Mullet', 'Trout']

unique_labels = sorted(set(labels))
label_encoder = {label: idx for idx, label in enumerate(unique_labels)}

model = load_model(save_path)

test = create_dataset(x_test)

pred = model.predict(test)
pred = np.argmax(pred, axis=1)

# Reverse mapping from label encoder
reverse_label_encoder = {idx: label for label, idx in label_encoder.items()}

# Convert prediction from encoded to class name
pred_class_name = reverse_label_encoder[pred[0]]
print(pred_class_name)