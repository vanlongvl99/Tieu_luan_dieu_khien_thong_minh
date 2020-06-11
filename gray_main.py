import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from os import listdir
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
import datetime


path = "data"
label_names = {}
y_labels = []
x_data = []

for forder_name in listdir(path):
    if forder_name.isdigit():
        label_names[forder_name] = int(forder_name)
    elif forder_name.lower() == "a":
        label_names[forder_name] = 10
    elif forder_name.lower() == "b":
        label_names[forder_name] = 11
    elif forder_name.lower() == "c":
        label_names[forder_name] = 12
    elif forder_name.lower() == "d":
        label_names[forder_name] = 13
    elif forder_name.lower() == "e":
        label_names[forder_name] = 14
print(label_names)

for forder_name in listdir(path):
    for file_name in listdir(path + "/" + forder_name):
        image = cv2.imread(path + "/" + forder_name +"/" + file_name)
        
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
        # print("shape:",image.shape)
        # exit()
        # print(image)
        image = (image/255)           #normalize
        image = cv2.resize(image, (64, 64))
        image = image.reshape(64,64,1)
        y_labels.append(label_names[forder_name])
        x_data.append(image)

print("=========")
print()
uni_labels = set(y_labels)
print(uni_labels)
x_data = tf.convert_to_tensor(x_data)
# print(x_data.shape)
# print(type(x_data[0]))
y_labels = tf.convert_to_tensor(y_labels)
# x_data = np.array(x_data)
# y_labels = np.array(y_labels)

print(type(x_data))

X_train, X_test, y_train, y_test = train_test_split(x_data.numpy(), y_labels.numpy(), test_size=0.05)

# print(len(type(X_train)))
print("============")

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(y_train[:10])

#Converting the labels into one hot encoding
#Building the model


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(20, 3, activation='relu'),
    tf.keras.layers.Conv2D(30, 3, activation='relu'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(30, 5, activation='relu'),
    tf.keras.layers.Conv2D(20, 3, activation='relu'),
    tf.keras.layers.Dropout(0.25),

    # tf.keras.layers.Conv2D(30, 3, activation='relu'),
    tf.keras.layers.Conv2D(30, 5, activation='relu'),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    # TODO: fill suitable activations
    tf.keras.layers.Dense(units=50, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=40, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=30, activation='relu'),

    tf.keras.layers.Dense(units=15, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
            optimizer = tf.keras.optimizers.Adam(),
            metrics = ['accuracy'])


print(datetime.datetime.now())
print()
model.fit(X_train, y_train, epochs=10, batch_size = 64)
# serialize weights to HDF5
print(datetime.datetime.now())
print()

model.save_weights("gray_model_weights_character6.h5")
model.save("gray_model_character6.h5")
print(model.evaluate(X_test,y_test))
# model.load_weights("model.h5")
