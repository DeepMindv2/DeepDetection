# Global Variables


CD = 'cd /Users/ryanjoseph/Desktop/Yale/Repositories/Pneumonia/Multi_Chest/Scripts'

# Dataset
MULTI_CHEST = True


# Paths
DATA_DIR_TRAIN = '/Users/ryanjoseph/Desktop/Yale/Repositories/Pneumonia/Multi_Chest/Data/chest_xray/train/'
DATA_DIR_VAL = '/Users/ryanjoseph/Desktop/Yale/Repositories/Pneumonia/Multi_Chest/Data/chest_xray/val/'
DATA_DIR_TEST = '/Users/ryanjoseph/Desktop/Yale/Repositories/Pneumonia/Multi_Chest/Data/chest_xray/test/'
LOG_DIR = '/Users/ryanjoseph/Desktop/Yale/Repositories/Pneumonia/Multi_Chest/Scripts/logs/'
WEIGHTS_DIR = '/Users/ryanjoseph/Desktop/Yale/Repositories/Pneumonia/Multi_Chest/Scripts/Weights/'
CLASS_NAMES = ['PNEUMONIA', 'NORMAL']

#Preprocessing
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_SHAPE = (IMG_HEIGHT,IMG_WIDTH,3)
TARGET_SIZE = (IMG_HEIGHT,IMG_WIDTH)


# Models
BIG_CNN = False
SMALL_CNN = False
DENSENET121 = True
DENSENET169 = False
VGG16 = False


# Model HyperParameters
BATCH_SIZE = 333
EPOCHS = 2
STEPS_PER_EPOCH = IMG_HEIGHT // BATCH_SIZE
LR = 0.1
DECAY = 1e-5


CYCLICAL_LR = False
EARLY_STOPPING = False
OPTIMIZER = 'Adam'
LOSS = 'binary_crossentropy'
MODEL_FINAL_ACTIVATION = 'sigmoid'
BASE_MODEL_TRAINABLE = False

# Augmentation
DATA_AUGMENTATION = False
ROTATION_RANGE = 45
ZCA_WHITENING = False
ZOOM_RANGE = 0.5
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
WIDTH_SHIFT_RANGE = .15
HEIGH_SHIFT_RANGE = .15

# Metrics
METRICS = ['accuracy', 'AUC', 'Precision', 'Recall']
SEED = 3

# Visualization
PLOT_LEARNING = False
PLOT_TESTING = False
MODEL = f'DenseNet121'
TENSORBOARD = 'tensorboard --logdir=logs/'


# weights
LOAD_WEIGHTS = True


# Hyper Parameter Optimization
IMG_SHAPE_OPT = (64,64)
OPTIMIZER_OPT = ['RMSprop', 'Adam', 'SGD', 'Adagrad', 'Nadam', 'Adamax']
LOSS_OPT = ['binary_crossentropy', 'categorical_crossentropy']
MODEL_FINAL_ACTIVATION_OPT = ['sigmoid', 'relu', 'softmax']

EXECUTIONS_PER_TRIAL = 5
MAX_TRIALS = 20
















#
