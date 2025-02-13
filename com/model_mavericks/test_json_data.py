import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Find an image of the digit 5
digit_5_idx = np.where(y_train == 6)[0][0]  # Get the first index where the label is 5
digit_5_image = X_train[digit_5_idx]

# Normalize and flatten the image
digit_5_image_normalized = digit_5_image.reshape(-1, 784) / 255.0
#print(f"digit_5_image_normalized {digit_5_image_normalized}")
# Convert to JSON format
test_data = {"features": digit_5_image_normalized.tolist()}
#print(f"test data {test_data}")
# Print the test data as JSON
import json
#print(json.dumps(test_data, indent=2))

# Save to a JSON file
file_path = 'mnist_digit_6_test_data.json'
with open(file_path, 'w') as json_file:
    json.dump(test_data, json_file, indent=2)

print(f"Saved test data to {file_path}")