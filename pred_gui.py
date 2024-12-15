import os
# Suppress oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Constants
IMG_HEIGHT_MOBILE = 224
IMG_WIDTH_MOBILE = 224
IMG_HEIGHT_CNN = 224
IMG_WIDTH_CNN = 224
IMG_HEIGHT_ANN = 64 
IMG_WIDTH_ANN = 64   
BATCH_SIZE = 1

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
def resize_and_pad_image(image, target_width = 590, target_height = 445, padding_color=(0, 0, 0)):
    # Get original image size
    original_width, original_height = image.size

    # Calculate aspect ratio and resize the image accordingly
    aspect_ratio = original_width / original_height
    if original_width > original_height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize the image while maintaining the aspect ratio
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a new image with the target size and fill it with the padding color
    padded_image = Image.new("RGB", (target_width, target_height), padding_color)

    # Calculate the position to paste the resized image
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # Paste the resized image onto the padded image
    padded_image.paste(resized_image, (paste_x, paste_y))

    return padded_image

# Function to predict the class of the image using all models
def predict_image_class(image_path):
    # Open the image and resize & pad it to 590x44
    image = Image.open(image_path)
    padded_image = resize_and_pad_image(image)

    # Convert the padded image to numpy array for each model

    # Prepare data for MobileNetV2: resize to 224x224
    img_array_mobile = np.array(padded_image.resize((IMG_WIDTH_MOBILE, IMG_HEIGHT_MOBILE), Image.Resampling.LANCZOS))
    img_array_mobile = np.expand_dims(img_array_mobile, axis=0)  # Add batch dimension
    mobile_data = preprocess_input(img_array_mobile)  # MobileNetV2 preprocessing

    # Prepare data for CNN: resize to 224x224
    img_array_cnn = np.array(padded_image.resize((IMG_WIDTH_CNN, IMG_HEIGHT_CNN), Image.Resampling.LANCZOS))
    img_array_cnn = np.expand_dims(img_array_cnn, axis=0)  # Add batch dimension
    cnn_data = img_array_cnn / 255.0  # Normalize for CNN

    # Prepare data for ANN: resize to 64x64
    img_array_ann = np.array(padded_image.resize((IMG_WIDTH_ANN, IMG_HEIGHT_ANN), Image.Resampling.LANCZOS))
    img_array_ann = np.expand_dims(img_array_ann, axis=0)  # Add batch dimension
    ann_data = img_array_ann / 255.0  # Normalize for ANN

    # Predictions from all models
    mobile_pred = mobile.predict(mobile_data)
    cnn_pred = cnn.predict(cnn_data)
    ann_pred = ann.predict(ann_data)

    # Decode predictions
    mobile_class = reverse_label_encoder[np.argmax(mobile_pred)]
    cnn_class = reverse_label_encoder[np.argmax(cnn_pred)]
    ann_class = reverse_label_encoder[np.argmax(ann_pred)]
    
    return mobile_class, cnn_class, ann_class

# GUI Application
def browse_image():
    global image_path, img_label, pred_label_mobile_result, pred_label_cnn_result, pred_label_ann_result

    # Open file dialog
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not image_path:
        return

    # Display the selected image
    img = Image.open(image_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk, width=300, height=300)
    img_label.image = img_tk

    # Predict and display the class for all models
    mobile_class, cnn_class, ann_class = predict_image_class(image_path)
    pred_label_mobile_result.config(text=mobile_class)
    pred_label_cnn_result.config(text=cnn_class)
    pred_label_ann_result.config(text=ann_class)

# Initialize the main window
root = tk.Tk()
root.title("Image Classification")
root.geometry("600x600")

# Create and place widgets
browse_button = tk.Button(root, text="Choose Image", command=browse_image)
browse_button.pack(pady=20)

img_frame = tk.Frame(root, width=300, height=300, bg="gray")
img_frame.pack(pady=20)
img_label = tk.Label(img_frame)
img_label.place(relx=0.5, rely=0.5, anchor="center")

# Labels to display the model names
name_label = tk.Label(root, text='Prediction', font=("Arial", 14))
name_label.pack(pady=5)

model_label_frame = tk.Frame(root)
model_label_frame.pack()

pred_label_mobile = tk.Label(model_label_frame, text="MobileNetV2: ", font=("Arial", 14))
pred_label_mobile.grid(row=0, column=0, sticky='w')
pred_label_mobile_result = tk.Label(model_label_frame, text="", font=("Arial", 14))
pred_label_mobile_result.grid(row=0, column=1, sticky='w')

pred_label_cnn = tk.Label(model_label_frame, text="CNN: ", font=("Arial", 14))
pred_label_cnn.grid(row=1, column=0, sticky='w')
pred_label_cnn_result = tk.Label(model_label_frame, text="", font=("Arial", 14))
pred_label_cnn_result.grid(row=1, column=1, sticky='w')

pred_label_ann = tk.Label(model_label_frame, text="ANN: ", font=("Arial", 14))
pred_label_ann.grid(row=2, column=0, sticky='w')
pred_label_ann_result = tk.Label(model_label_frame, text="", font=("Arial", 14))
pred_label_ann_result.grid(row=2, column=1, sticky='w')

# Run the GUI event loop
root.mainloop()
