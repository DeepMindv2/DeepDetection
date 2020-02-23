# Necessary Dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation
from tensorflow.keras.models import Model, Sequential
import kerastuner
from kerastuner.tuners import RandomSearch

from DataAug import *
from configs import *

#[From Sentdex Video]

def tuning_model(hp):
    model = keras.models.Sequential()

    model.add(Conv2D(hp.Int('input_units',
                                min_value=32,
                                max_value=256,
                                step=32), (3, 3), input_shape=(160,160,3)))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(hp.Int('n_layers', 1, 4)):  # adding variation of layers.
        model.add(Conv2D(hp.Int(f'conv_{i}_units',
                                min_value=32,
                                max_value=256,
                                step=32), (3, 3)))
        model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(9))
    model.add(Activation("softmax"))


    model.summary()
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=METRICS)

    return model
# done

tuner = RandomSearch(
        tuning_model,
        objective = 'val_accuracy',
        max_trials = 2,
        executions_per_trial = 2, # Training same model X times
        directory = LOG_DIR
)

tuner.search_space_summary()


tuner.search(train_data_gen,
            epochs=EPOCHS,
            validation_data=val_data_gen)


tuner.results_summary()




























#
