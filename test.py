#run in interactive mode to see plot output, or highlight the code and press Shift+Enter

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# Define the dataset directory
fish_dir = 'Fish_Dataset'

# Initialize lists to store paths and labels
path = []
label = []

# Walk through the dataset directory
for dir_name, _, filenames in os.walk(fish_dir):
    for filename in filenames:
        if filename.endswith('.png') and 'GT' not in dir_name:
            folder_name = dir_name.split(os.sep)[-1]
            folder_name = folder_name.replace('Hourse Mackerel', 'Horse Mackerel')
            folder_name = folder_name.replace('Gilt-Head Bream', 'Gilt Head Bream')
            label.append(folder_name)
            path.append(os.path.join(dir_name, filename))

# Create a DataFrame with paths and labels
data = pd.DataFrame({'path': path, 'label': label})
data['label'] = data['label'].replace('Hourse Mackerel', 'Horse Mackerel')
data['label'] = data['label'].replace('Gilt-Head Bream', 'Gilt Head Bream')

# Plot the images
plt.figure(figsize=(15, 12))
idx = 0
for unique_label in data['label'].unique():
    plt.subplot(3, 3, idx + 1)
    plt.imshow(plt.imread(data[data['label'] == unique_label].iloc[0, 0]))
    plt.title(unique_label)
    plt.axis('off')
    idx += 1
plt.show()