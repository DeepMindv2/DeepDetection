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
from Visualization import *



def tensorboard():
    print('\n-------- Starting Up Tensorboard! --------\n')
    run = f"{MODEL}-Augmentation_{DATA_AUGMENTATION}-Optimizer_{OPTIMIZER}-LR_{LR}-Loss_{LOSS}-Img_Shape_{IMG_SHAPE}-Time_{time.localtime().tm_min}"
    #run = '{}-{}-{}'.format(MODEL,DATA_AUGMENTATION, OPTIMIZER, time.localtime().tm_min)
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


    #tf.summary.scalar('learning rate', data=lr_schedule, step=epoch)

    return lr_schedule

def ReduceLROnPlateau_Callback():
    # This is a callback, can't put this on the optimizer itself
    print('\n-------- Reduce Learning Rate on Plateau --------\n')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                factor=0.1, # How fast learning rate will decrease
                                patience=1,
                                cooldown=1, # Number of Epochs with no improvements
                                verbose=2)

    #tf.summary.scalar('learning rate', data=reduce_lr, step=epoch)

    return reduce_lr
