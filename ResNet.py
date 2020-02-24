# Necessary Dependencies
import tensorflow as tf
import time
import scipy
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications import ResNet101
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from configs import *
from DataAug import *
from Visualization import *
from Utils import *

# Fixed a bug that made us unable to download the pre-trained weights of models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


############################## ResNet 101 ##############################

def train_ResNet101():
    if ResNet101:
        print('\nTraining Model -- ResNet 101\n')

        # Pre-trained on ImageNet
        # Default input size is 224x224

        base_model = tf.keras.applications.ResNet101(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = BASE_MODEL_TRAINABLE

        x = base_model.output

        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid',name='Final')(x)


        model = Model(inputs=base_model.input, outputs=predictions)


        load_weights(model)

        model.compile(loss=LOSS,
                          optimizer=OPTIMIZER,
                          metrics=METRICS)

        #print(model.summary())

        history = model.fit (
            train_data_gen,
            steps_per_epoch = 5216 // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data = val_data_gen,
            validation_steps = 624 // BATCH_SIZE,
            callbacks=[save_weights_callback(), tensorboard(), ReduceLROnPlateau_Callback()]
        )


        save_weights(model)

        if PLOT_LEARNING:
            plot_learning_curve(history)

        score = model.evaluate(test_data_gen)
        print(f'Model Test Loss: {score[0]}')
        print(f'Model Test Accuracy: {score[1]}')

        ConfusionMatrix(model, test_data_gen)

        return history





train_ResNet101()
