import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split


class1 = []
class2 = []

folder = r"C:\Users\toivo\OneDrive - TUNI.fi\Kurssimateriaalit\DATA.ML.200 Pattern Recognition and Machine Learning\Exercises\Ex5\GTSRB_subset_2\class1"
for filename in os.listdir(folder):
    class1.append(cv2.imread(os.path.join(folder, filename)))

folder = r"C:\Users\toivo\OneDrive - TUNI.fi\Kurssimateriaalit\DATA.ML.200 Pattern Recognition and Machine Learning\Exercises\Ex5\GTSRB_subset_2\class2"
for filename in os.listdir(folder):
    class2.append(cv2.imread(os.path.join(folder, filename)))

class1 = np.array(class1)
class2 = np.array(class2)

gt_class1 = 0*np.ones(class1.shape[0])
gt_class2 = 1*np.ones(class2.shape[0])

x = np.vstack((class1, class2))
y = np.hstack((gt_class1, gt_class2))

#x = x/255

y = tf.keras.utils.to_categorical(y, 10)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


model = tf.keras.models.Sequential()

# Flattening the input image
model.add(keras.layers.Flatten(input_shape=(64, 64, 3)))

# Fully connected layer 1
model.add(keras.layers.Dense(100, activation='sigmoid'))

# Fully connected layer 2
model.add(keras.layers.Dense(100, activation='sigmoid'))

# Output layer
model.add(keras.layers.Dense(10, activation='sigmoid'))

# Defining the loss function and compiling the model
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True),
model.compile(optimizer='SGD', loss=loss_fn, metrics=['accuracy'])

model.summary()

# Training the model
model.fit(X_train, y_train, epochs=10)

print("Training done")

#y_test_hat = model.predict(X_test)
model.evaluate(X_test, y_test, verbose=2)
