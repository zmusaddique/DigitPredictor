from sklearn.datasets import fetch_openml
import numpy as np

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Separate the test set images and labels
test_images = mnist.data[60000:]
test_labels = mnist.target[60000:].astype(np.uint8)

# Save the test set images and labels as NumPy arrays
np.save('test_images.npy', test_images)
np.save('test_labels.npy', test_labels)
