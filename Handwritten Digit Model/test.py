from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load the trained model
model = load_model("m1_ai_model.h5")

import numpy as np

# Load test data (optional, if you want to test the model)
#(_, _), (x_test, y_test) = mnist.load_data()

# Preprocess the new data (normalize and reshape)
#x_test = x_test / 255.0  # Normalize
#x_test = x_test.reshape(-1, 28, 28)  # Ensure correct shape

# Select a sample image (e.g., first test image)
  # Reshape for prediction

import numpy as np

# Load MNIST dataset from the downloaded file
with np.load("mnist.npz") as data:
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

# Normalize the images (optional, but recommended)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape if your model expects a specific input shape
x_train = x_train.reshape(-1, 28, 28)  # Example for a model expecting 28x28 images
x_test = x_test.reshape(-1, 28, 28)

print(f"Training data shape: {x_train.shape}, Labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, Labels shape: {y_test.shape}")

sample_image = x_test[0].reshape(1, 28, 28)
# Get the model's predictions
predictions = model.predict(sample_image)

# Convert predictions to class labels
predicted_label = np.argmax(predictions)

print(f"Predicted Digit: {predicted_label}")


plt.imshow(x_test[0], cmap="gray")
plt.title(f"Predicted Label: {predicted_label}")
plt.show()
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
