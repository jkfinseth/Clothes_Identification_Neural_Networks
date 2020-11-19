# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Import necessary data
data = keras.datasets.fashion_mnist

# Seperate data points into training and tests
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Define what each image could be
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Change images to a scale [0,1]
train_images = train_images/255.0
test_images = test_images/255.0

# Flatten images
train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))

# Define the model
# First layer is the input layer, takes an array of 784 numbers
# Second layer is the "hidden layer" that allows calculations to occur
# Third layer is the output layer
model = keras.models.Sequential()
model.add(keras.layers.Dense(128, activation='relu', input_dim=784))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# Train the model to adjust the hidden layers to raise accuracy
# epochs = number of repititions
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("\nTest accuracy: " + str(test_acc))
