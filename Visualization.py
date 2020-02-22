# Necessary Dependencies
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from configs import *



def plot_learning_curve(history):
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


def plot_testing_curve(history):
    acc = history.history['accuracy']

    loss=history.history['loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.legend(loc='lower right')
    plt.title('Testing Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.legend(loc='upper right')
    plt.title('Testing Loss')
    plt.show()


def tensorboard():
    print('\n-------- Starting Up Tensorboard! --------\n')

    run = '{}-{}-{}'.format(MODEL, time.localtime().tm_hour, time.localtime().tm_min)
    tensorboard_logs= LOG_DIR + run
    print(f'Saving Tensorboard file in {tensorboard_logs}')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs, histogram_freq=1)
    return tensorboard_callback



def Early_Stopping():
    print('\n-------- Initializing Early Stopping --------\n')
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', # When this goes down stop training
        min_delta=1e-2, # Learning Rate Comparison
        patience=3, # No improvement for ____ more epochs
        verbose=2
        )
    return early_stop


def load_weights(model):
    print('\n-------- Getting Pre-trained Weights --------\n')
    weights = model.get_weights()
    model.set_weights(weights)
    return model

def save_weights(model):
    print('\n-------- Saving Trained Weights --------\n')
    model.save_weights(WEIGHTS_DIR)
    return model


def save_weights_callback():
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=WEIGHTS_DIR,
                                                monitor='val_accuracy',
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1)
    return cp_callback


def CSVLOGGER():
    print('Saving all data to CSV')
    #data_aug = 'Data Augmentation: ' + DATA_AUGMENTATION




def ExponentialDecayLR():
    print('\n-------- Exponential Decay Learning Rate --------\n')
    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    return lr_schedule

def ReduceLROnPlateau_Callback():
    # This is a callback, can't put this on the optimizer itself
    print('\n-------- Reduce Learning Rate on Plateau --------\n')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                factor=0.1, # How fast learning rate will decrease
                                patience=1,
                                cooldown=1, # Number of Epochs with no improvements
                                verbose=2)

    return reduce_lr


def ConfusionMatrix(model, test_data_gen):
    print('Building Confusion Matrix')
    preds = model.predict(test_data_gen)
    preds = np.squeeze((preds > 0.5).astype('int'))
    orig = test_data_gen.classes
    cm  = confusion_matrix(orig, preds)
    plt.figure()
    plot_confusion_matrix(cm,figsize=(30,15), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=26)
    plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=26)
    plt.show()





















#
