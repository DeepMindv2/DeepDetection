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

# HyperParameters


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
model_2 = Sequential([
    Conv2D(64, 3, padding='same', activation='relu', input_shape=(160, 160 ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(62, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_2.summary()

model.load_weights('Weights/')

model_2.compile(optimizer=OPTIMIZER,
              loss=LOSS,
              metrics=METRICS)

history = model_2.fit_generator(
    train_data_gen,
    steps_per_epoch=STEPS_EPOCH,
    epochs=EPOCHS,
    validation_data = val_data_gen,
    callbacks=[tensorboard_callback, csv_logger]
)


# Save Weights
#model.save_weights('Weights/')







# Visualization
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
