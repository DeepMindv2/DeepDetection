# Necessary Dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import CSVLogger

from configs import *
from DataAug import *



### Data Augmentation taking place in seperate folder


###



# Learning Model
if BIG_CNN:
    print('Starting to Train Testing CNN')
    model_2 = Sequential([
        Conv2D(124, 3, padding='same', activation='relu', input_shape=(160, 160 ,3)),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(62, activation='relu'),
        Dense(1, activation=MODEL_FINAL_ACTIVATION)
    ])

if SMALL_CNN:
    print('Starting to Train Dense_CNN')
    model_2 = Sequential([
        Conv2D(64, 3, padding='same', activation='relu', input_shape=(160, 160 ,3)),
        MaxPooling2D(),
        Flatten(),
        Dense(1, activation=MODEL_FINAL_ACTIVATION)
    ])


model_2.summary()


model_2.compile(optimizer=OPTIMIZER,
              loss=LOSS,
              metrics=METRICS)

history = model_2.fit_generator(
    train_data_gen,
    steps_per_epoch=STEPS_EPOCH,
    epochs=EPOCHS,
    validation_data = val_data_gen
)
