# Necessary Dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201

import kerastuner
from kerastuner.tuners import RandomSearch, BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner.engine.hypermodel import HyperModel

from DataAug import *
#from DenseNet import *
from configs import *

hp = HyperParameters()
OPTIMIZER_OPT = hp.Choice('optimizer', values=['Adam', 'SGD', 'RMSprop'], default='Adam')


def train_DenseNet169(hp):
    if DENSENET169:
        print('Training Model -- DenseNet 169')
        print('Steps Per Epoch: ', 5216//BATCH_SIZE)

        print('\n---------------------- HyperParameter Optimization ----------------------\n')

        base_model = tf.keras.applications.DenseNet169(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')


        #base_model.summary()
        base_model.trainable = BASE_MODEL_TRAINABLE

        x = base_model.output

        x = GlobalAveragePooling2D()(x)
        predictions = Dense(1, activation='sigmoid',name='Final')(x)



        model = Model(inputs=base_model.input, outputs=predictions)


        load_weights(model)


        model.compile(loss=LOSS,
                          optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                          metrics=METRICS)

        return model


b_tuner = BayesianOptimization(
    train_DenseNet169,
    objective='val_accuracy',
    max_trials=MAX_TRIALS,
    executions_per_trial=EXECUTIONS_PER_TRIAL,
    directory=LOG_DIR,
    project_name='BAYES_OPTIMIZERS',
    seed=SEED
)

b_tuner.search_space_summary()

b_tuner.search(train_data_gen,
            epochs=EPOCHS,
            validation_data=val_data_gen)

b_tuner.results_summary()

b_tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values

tuner.get_best_models()[0]



train_DenseNet169(hp)


































#
