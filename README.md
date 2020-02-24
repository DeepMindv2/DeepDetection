# DeepDetection_2
Pneumonia detection using a variety of Convolutional Neural Networks. Bayesian Classifiers will also be tested. 

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
- keras_tuner.py # Bayesian Optimization of Optimizer and Loss Functions 
- Visualization.py # Various visualization functions 
- EDA2.py # Inspection of Dataset 

```

# Data
The Pneumonia dataset was from Kaggle. The dataset contained 5,863 X-ray images. The dataset contained only 2 categories, Normal and Pneumonia. The dataset is roughly 3 GB in size. 

The X-ray images are from patients 1-5 years old. All chest x-ray images were taken Anterior and Posterior. 

The images come from a hospital in China, Guangzhou Women and Children’s Medical Center. 

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

![Pneumonia](https://github.com/DeepMindv2/DeepDetection/blob/master/Screenshots/Screen%20Shot%202020-02-22%20at%202.07.16%20PM.png)
![Normal](https://github.com/DeepMindv2/DeepDetection/blob/master/Screenshots/Screen%20Shot%202020-02-22%20at%202.06.51%20PM.png)


# Exploratory Data Analysis
A large class imbalance on the original dataset was noticed, in the train and test folders. Extensive data augmentation was needed to get accurate and reliable results. The images come in varying sizes. Of the 4 types of Pneumonia the dataset consisted of only Bacterial and Viral Pneumonia. 
```
EDA2.ipynb
```
![Training Class Imbalance](https://github.com/DeepMindv2/DeepDetection/blob/master/Screenshots/Screen%20Shot%202020-02-22%20at%202.14.26%20PM.png)
![Testing Class Imbalance](https://github.com/DeepMindv2/DeepDetection/blob/master/Screenshots/Screen%20Shot%202020-02-22%20at%202.14.44%20PM.png)

# Data Augmentation
Implented a cohort of Data Augmentation strategies. I found it best to just use all of them. Using the Keras-Tuner library I did a Random Search as well as a Bayesian Optimization to see which if not all Augmentation strategies yield the highest accuracies. 

- Rescale = 1./255
- Rotation Range = 45
- ZCA Whitening
- Zoom Range = 0.5
- Horizontal Flip
- Vertical Flip
- Width Shift Range = .15
- Height Shift Range = .15

```
python3 DataAug.py
python3 Keras_Tuner.py
```

# Software
- Tensorflow 2.1
- Keras-Tuner
- Matplotlib
- Scikit Learn 
- Scipy
- mlxtend
- kerastuner

# Model
- DenseNet
```
python3 DenseNet.py
```
- ResNet
```
python3 ResNet.py
```
- VGG
```
VGG.py
```

# Evaluation
Due to compute issues I was not able to train all the networks. I was also not able to implement a wide Bayesian Optimization search. Very little training was able to be done on my own. I had to cycle in between training locally on my laptop and Colab. In the future, I'm going to solve this by having a smaller proxy dataset and I am going to do all my preprocessing and Data Augmentation before my training, this was severely hampering the training time. 

![Training Logs](https://github.com/DeepMindv2/DeepDetection/blob/master/Screenshots/Screen%20Shot%202020-02-23%20at%208.58.18%20PM.png)

```
python3 Visualization.py
```

# Next Steps
- Implementing Class Activation Maps to see where in the X-ray image the CNN is looking towards.
- Developing an ensemble of these CNN models 
- Creating a cache method to store dataset to speed up training of models. Pre-processing and Data Augmentation while training heavily relies on the CPU which takes away from the GPU benefits
- Implenting a Focal Loss, more suitable for imbalanced datasets
- Label Smoothing (LabelSmoothingCrossEntropy())
- Batch Normalization
- Test Rectified Adam, it slowly reduces the learning rate until the variance stabilizes itself
- Cosine Anneal Learning Rate, this increases and decreases the learning rate which helps avoid(jump) smaller local optimas
- Figuring out the image size that stores enough information while remaining the small
- Adding an attention layer would be super interesting, they allow NN to capture longer range dependencies and focus in on more object shapes
- Implementing more data augmentation strategies I believe is key, more important than the CNN architecture and dropout. Based on this paper, https://arxiv.org/abs/1806.03852v4,
- New Data Augmentation technique, Progressive Sprinkling. Outperforms all other techniques: Cutmix, Mixup, and Recap. It's just partial masking of certain areas of the image. It forces the network to seek more relevant areas of interest.  
- Curating and smaller proxy dataset that is representative of the entire dataset, for quick and rapid hyperparameter testing. Based on, https://arxiv.org/abs/1906.04887v1.(it would also be more balanced)
- Implementing EfficentNet, designed by google, have some techniques I can add to existing NN architectures (DenseNet)





