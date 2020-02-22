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

print('Started')
#start = time.time()

# HyperParameters
MODEL_NAME = 'Pretrained_CNN-'
BATCH_SIZE = 15
EPOCHS = 5
STEPS_EPOCH = 15
LR = 0.0001
METRICS = ['accuracy','binary_crossentropy', tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()]
SEED = 2
IMG_SIZE = 160
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

data_dir_train = '/Users/ryanjoseph/Desktop/Yale/Repositories/Pneumonia/Multi_Chest/Data/chest_xray/train/'
data_dir_val = '/Users/ryanjoseph/Desktop/Yale/Repositories/Pneumonia/Multi_Chest/Data/chest_xray/val/'

CLASS_NAMES = ['Pneumonia', 'Normal']


image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                                                    rescale=1./255,
                                                    rotation_range=45,
                                                    zoom_range=0.5,
                                                    horizontal_flip=True,
                                                    width_shift_range=.15,
                                                    height_shift_range=.15,
                                                    )

validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir_train),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     seed=SEED,
                                                     target_size=(160, 160),
                                                     classes = list(CLASS_NAMES))

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=str(data_dir_val),
                                                              seed=SEED,
                                                              target_size=(160, 160),
                                                              classes = list(CLASS_NAMES))




# CSV Logger
csv_logger = CSVLogger('training.log.csv')


# Tensorboard
tensorboard_logs="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs, histogram_freq=1)

# Learning Model
base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

#base_model.summary()
base_model.trainable = True
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='relu',name='Final')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=METRICS)

print(model.summary())

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=STEPS_EPOCH,
    epochs=EPOCHS,
    validation_data = val_data_gen
    #callbacks=[tensorboard_callback, csv_logger]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()








































#
