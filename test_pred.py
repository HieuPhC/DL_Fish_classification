import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Suppress oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
save_path = "saved_models/MobileNetV2.keras"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 1

# Preprocessing function
def preprocess_image_with_padding_tensor(image, target_height=IMG_HEIGHT, target_width=IMG_WIDTH):
    padded_img = tf.image.resize_with_pad(image, target_height, target_width)
    return padded_img

# Load model
model = load_model(save_path)

# Define labels and reverse label encoding
labels = ['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel',
          'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp',
          'Striped Red Mullet', 'Trout']

unique_labels = sorted(set(labels))
label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
reverse_label_encoder = {idx: label for label, idx in label_encoder.items()}

# Function to predict the class of the image
def predict_image_class(image_path):
    test_data_generator = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_image_with_padding_tensor)
    
    data = pd.DataFrame({'path': [image_path]})
    test = test_data_generator.flow_from_dataframe(
        dataframe=data, x_col='path', target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='rgb', class_mode='input', batch_size=BATCH_SIZE, shuffle=False)

    pred = model.predict(test)
    pred = np.argmax(pred, axis=1)
    return reverse_label_encoder[pred[0]]

# GUI Application
def browse_image():
    global image_path, img_label, pred_label

    # Open file dialog
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not image_path:
        return

    # Display the selected image
    img = Image.open(image_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk, width=300, height=300)
    img_label.image = img_tk

    # Predict and display the class
    predicted_class = predict_image_class(image_path)
    pred_label.config(text=f"Predicted Class: {predicted_class}")

# Initialize the main window
root = tk.Tk()
root.title("Image Classification")
root.geometry("600x500")

# Create and place widgets
browse_button = tk.Button(root, text="Choose Image", command=browse_image)
browse_button.pack(pady=20)

img_frame = tk.Frame(root, width=300, height=300, bg="gray")
img_frame.pack(pady=20)
img_label = tk.Label(img_frame)
img_label.place(relx=0.5, rely=0.5, anchor="center")

pred_label = tk.Label(root, text="Predicted Class: ", font=("Arial", 18))
pred_label.pack(pady=20)

# Run the GUI event loop
root.mainloop()