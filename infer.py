import os
# Suppress warnings and logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# Constants
IMG_HEIGHT_MOBILE = 224
IMG_WIDTH_MOBILE = 224
IMG_HEIGHT_CNN = 224
IMG_WIDTH_CNN = 224
IMG_HEIGHT_ANN = 64 
IMG_WIDTH_ANN = 64   

# Load models
mobile = load_model("saved_models/MobileNetV2.keras")
cnn = load_model("saved_models/cnn_1_model.keras")
ann = load_model("saved_models/ann_model.keras")

# Define labels and reverse label encoding
labels = ['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel',
          'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp',
          'Striped Red Mullet', 'Trout']

unique_labels = sorted(set(labels))
label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
reverse_label_encoder = {idx: label for label, idx in label_encoder.items()}

# Function to resize and pad the image to the target size
def resize_and_pad_image(image, target_width=590, target_height=445, padding_color=(0, 0, 0)):
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    if original_width > original_height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    padded_image = Image.new("RGB", (target_width, target_height), padding_color)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    padded_image.paste(resized_image, (paste_x, paste_y))
    return padded_image

# Function to predict the class of the image using all models
def predict_image_class(image_path):
    image = Image.open(image_path)
    padded_image = resize_and_pad_image(image)

    # Prepare data for MobileNetV2
    img_array_mobile = np.array(padded_image.resize((IMG_WIDTH_MOBILE, IMG_HEIGHT_MOBILE), Image.Resampling.LANCZOS))
    img_array_mobile = np.expand_dims(img_array_mobile, axis=0)
    mobile_data = preprocess_input(img_array_mobile)

    # Prepare data for CNN
    img_array_cnn = np.array(padded_image.resize((IMG_WIDTH_CNN, IMG_HEIGHT_CNN), Image.Resampling.LANCZOS))
    img_array_cnn = np.expand_dims(img_array_cnn, axis=0)
    cnn_data = img_array_cnn / 255.0

    # Prepare data for ANN
    img_array_ann = np.array(padded_image.resize((IMG_WIDTH_ANN, IMG_HEIGHT_ANN), Image.Resampling.LANCZOS))
    img_array_ann = np.expand_dims(img_array_ann, axis=0)
    ann_data = img_array_ann / 255.0

    # Predictions from all models
    mobile_pred = mobile.predict(mobile_data)
    cnn_pred = cnn.predict(cnn_data)
    ann_pred = ann.predict(ann_data)

    # Decode predictions
    mobile_class = reverse_label_encoder[np.argmax(mobile_pred)]
    cnn_class = reverse_label_encoder[np.argmax(cnn_pred)]
    ann_class = reverse_label_encoder[np.argmax(ann_pred)]

    return mobile_class, cnn_class, ann_class

# Main function to handle command-line arguments and output predictions
def main():
    parser = argparse.ArgumentParser(description="Image Classification Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()

    # Validate image path
    image_path = args.image_path
    if not os.path.exists(image_path):
        print(f"Error: The image file '{image_path}' does not exist.")
        return

    # Predict image class
    mobile_class, cnn_class, ann_class = predict_image_class(image_path)

    # Save predictions to a text file
    output_file = "predictions.txt"
    with open(output_file, "w") as file:
        file.write(f"Predictions for image: {image_path}\n")
        file.write(f"MobileNetV2 Prediction: {mobile_class}\n")
        file.write(f"CNN Prediction: {cnn_class}\n")
        file.write(f"ANN Prediction: {ann_class}\n")

    print()
    print(f"MobileNetV2 Prediction: {mobile_class}")
    print(f"CNN Prediction: {cnn_class}")
    print(f"ANN Prediction: {ann_class}")
    
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()