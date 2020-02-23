# Necessary Dependencies
import tensorflow as tf
import time
import scipy
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import Accuracy, AUC, Precision, SensitivityAtSpecificity, Recall
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201

from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from configs import *
from DataAug import *
from Visualization import *
from Utils import *


############################## DenseNet121 ##############################

def train_DenseNet121():
    if DENSENET121:
        print('\nTraining Model -- DenseNet 121\n')
        base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        #base_model.summary()
        base_model.trainable = BASE_MODEL_TRAINABLE

        x = base_model.output

        x = GlobalAveragePooling2D()(x)
        predictions = Dense(1, activation='sigmoid',name='Final')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        #load_weights(model)

        model.compile(loss=LOSS,
                          optimizer=OPTIMIZER,
                          metrics=METRICS)

        #print(model.summary())

        history = model.fit ( # Changed this from model.fit_generator
            val_data_gen,
            steps_per_epoch = 5, #5216 // BATCH_SIZE,
            epochs=EPOCHS,
            #validation_data = val_data_gen,
            #validation_steps = 624 // BATCH_SIZE,
            callbacks=[tensorboard()]
        )


        save_weights(model)

        if PLOT_LEARNING:
            plot_learning_curve(history)

        score = model.evaluate(test_data_gen)
        print(f'Model Test Loss: {score[0]}')
        print(f'Model Test Accuracy: {score[1]}')

        ConfusionMatrix(model, test_data_gen)



        return history


train_DenseNet121()


















#
