# DeepDetection_2
Pneumonia detection using a variety of Convolutional Neural Networks. Bayesian Classifiers are also tested. 

# Overview
The repository contains various Neural Network Architectures used in the training of Pnemonia classification. 

# Code Flow
```

- configs.py # Centralizes all hyperparameters for models and augmentations

- DenseNet.py # Various DenseNet Model definitions
- ResNet.py # Various ResNet Model definitions
- VGG.py # Various VGG Model definitions
- VanillaCNN.py # Simple Convolutional Neural Network

- DataAug.py # generator for train/test 
- keras_tuner.py # 
- Visualization.py # Various visualization functions 

```


# Data
The Pneumonia dataset was from Kaggle. The dataset contained 5,863 X-ray images. The dataset contained only 2 categories, Normal and Pneumonia. The dataset is roughly 3 GB in size. 

The X-ray images are from patients 1-5 years old. All chest x-ray images were taken Anterior and Posterior. 

The images come from a hospital in China, Guangzhou Women and Children’s Medical Center. 

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

![Pneumonia](https://github.com/DeepMindv2/DeepDetection/blob/master/Screen%20Shot%202020-02-22%20at%202.06.51%20PM.png)
![Normal](https://github.com/DeepMindv2/DeepDetection/blob/master/Screen%20Shot%202020-02-22%20at%202.07.16%20PM.png)


# Exploratory Data Analysis
A large class imbalance on the original dataset was noticed, in the train and test folders. Extensive data augmentation was needed to get accurate and reliable results. The images come in varying sizes. Of the 4 types of Pneumonia the dataset consisted of only Bacterial and Viral Pneumonia. 

![Training Class Imbalance](https://github.com/DeepMindv2/DeepDetection/blob/master/Screen%20Shot%202020-02-22%20at%202.14.26%20PM.png)
![Testing Class Imbalance](https://github.com/DeepMindv2/DeepDetection/blob/master/Screen%20Shot%202020-02-22%20at%202.14.44%20PM.png)

# Data Augmentation
Implented a cohort of Data Augmentation strategies. I found it best to just use all of them. Using the Keras-Tuner library I did a Random Search as well as a Bayesian Optimization to see which if not all Augmentation strategies yield the highest accuracies. 



# Software
- Tensorflow 2.1
- Keras-Tuner
- Matplotlib
- Scikit Learn 
- Scipy
- mlxtend
- kerastuner

# Hardware

# Model
- DenseNet
- ResNet
- 

# Heat Maps

# Evaluation

# References 

