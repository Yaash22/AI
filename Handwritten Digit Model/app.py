import tensorflow as tf
print(tf.__version__)

from tensorflow import keras
import os
from tensorflow.keras.datasets import mnist


# Load dataset
#mnist = keras.datasets.mnist
#mnist_path = os.path.expanduser("/Users/yashau/Documents/Web Development/AI/mnist.npz")
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
import numpy as np
with np.load("mnist.npz") as data:
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

model.save("m1_ai_model.h5")
# Load later
new_model = keras.models.load_model("m1_ai_model.h5")
