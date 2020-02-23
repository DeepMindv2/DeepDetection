# Necessary Dependencies
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from configs import *


if DATA_AUGMENTATION:
    print('Starting Data Augmentation!')
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                                                        rescale=1./255,
                                                        rotation_range=ROTATION_RANGE,
                                                        zca_whitening=ZCA_WHITENING,
                                                        zoom_range=ZOOM_RANGE,
                                                        horizontal_flip=HORIZONTAL_FLIP,
                                                        vertical_flip=VERTICAL_FLIP,
                                                        width_shift_range=WIDTH_SHIFT_RANGE,
                                                        height_shift_range=HEIGH_SHIFT_RANGE
                                                        )


elif DATA_AUGMENTATION==False:
    print('No Data Augmentation!')
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

validation_image_generator = ImageDataGenerator(rescale=1./255,
                                                rotation_range=ROTATION_RANGE,
                                                zca_whitening=ZCA_WHITENING,
                                                zoom_range=ZOOM_RANGE,
                                                horizontal_flip=HORIZONTAL_FLIP,
                                                vertical_flip=VERTICAL_FLIP,
                                                width_shift_range=WIDTH_SHIFT_RANGE,
                                                height_shift_range=HEIGH_SHIFT_RANGE)


test_image_generator = ImageDataGenerator(rescale=1./255)


train_data_gen = image_generator.flow_from_directory(directory=str(DATA_DIR_TRAIN),
                                                     shuffle=True,
                                                     seed=SEED,
                                                     target_size=TARGET_SIZE,
                                                     classes = list(CLASS_NAMES),
                                                     class_mode='binary'
                                                     )

val_data_gen = validation_image_generator.flow_from_directory(
                                                              directory=str(DATA_DIR_TEST),
                                                              shuffle=True,
                                                              seed=SEED,
                                                              batch_size = BATCH_SIZE,
                                                              target_size=TARGET_SIZE,
                                                              classes = list(CLASS_NAMES),
                                                              class_mode='binary'
                                                              )
test_data_gen = test_image_generator.flow_from_directory(
                                                              directory=str(DATA_DIR_VAL),
                                                              shuffle = True,
                                                              seed=SEED,
                                                              batch_size = BATCH_SIZE,
                                                              target_size=TARGET_SIZE,
                                                              classes = list(CLASS_NAMES),
                                                              class_mode='binary'
                                                              )









#
