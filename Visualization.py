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
