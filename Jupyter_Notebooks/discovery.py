# Import Necessary Dependencies
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import time
import os
from tqdm import tqdm
import tensorflow
import tensorflow as tf
#import scipy
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


print('\n######################## Script Started ########################\n')
start_preprocessing = time.time()

############################################ Pre-processing ############################################

labels_df = pd.read_csv('/Users/ryanjoseph/Desktop/Yale/Repositories/Pneumonia/Sample_ChestXRay_2GB/sample/sample_labels.csv')

print('# of Scans: ', len(labels_df))
labels_df.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                  'Patient_Age', 'Patient_Gender', 'View_Position',
                  'Original_Image_Width', 'Original_Image_Height',
                  'Original_Image_Pixel_Spacing_X',
                  'Original_Image_Pixel_Spacing_Y']
#print(labels_df.head(5))

# Figuring out how many patients we have
num_glob = glob('/Users/ryanjoseph/Desktop/Yale/Repositories/Pneumonia/Sample_ChestXRay_2GB/sample/sample/images/*.png')
print(f'There are {len(num_glob)} images')

# Mapping Images to their respective Paths
img_path = {os.path.basename(x): x for x in num_glob}
labels_df['Paths'] = labels_df['Image_Index'].map(img_path.get)
#print(labels_df.head())

# Diseases classified in the Stanford Paper
disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
Pneumonia_label = ['Pneumonia']

# One hot encoding all Diseases
for diseases in tqdm(disease_labels): #TQDM is a progress bar setting
    labels_df[diseases] = labels_df['Finding_Labels'].map(lambda result: 1.0 if diseases in result else 0)
#print(labels_df.head())

# Number of training samples for each diseases
num_disease = labels_df[disease_labels].sum().sort_values(ascending=False)
print(num_disease, [...])

# Vectorize all diseases
labels_df['All_Diseases_Targets'] = labels_df.apply(lambda target: [target[disease_labels].values],1).map(lambda target:target[0])
labels_df['Pneumonia_Target'] = labels_df.apply(lambda target: [target[Pneumonia_label].values],1).map(lambda target:target[0])


#train_set, test_set = train_test_split(xray_data, test_size = 0.2, random_state = 1993)
test_set = str(labels_df)


# How Long it took for Preprocessing
pre_time = (time.time() - start_preprocessing)/60
print(f'\nPreprocessing took {pre_time} seconds!\n')

df = labels_df

df.to_csv('New.csv')



#	Image_Index	Finding_Labels	Follow_Up_#	Patient_ID	Patient_Age	'Patient_Gender'	View_Position	Original_Image_Width	Original_Image_Height	Original_Image_Pixel_Spacing_X	Original_Image_Pixel_Spacing_Y	Paths	Atelectasis	Consolidation	Infiltration	Pneumothorax	Edema	Emphysema	Fibrosis	Effusion	Pneumonia	Pleural_Thickening	Cardiomegaly	Nodule	Mass	Hernia	All_Diseases_Targets	Pneumonia_Target

DROP = ['Finding_Labels', 'Follow_Up_#', 'Patient_ID', 'Patient_Age', 'Patient_Gender', 'View_Position', 'Original_Image_Width', 'Original_Image_Height', 'Original_Image_Pixel_Spacing_X', 'Original_Image_Pixel_Spacing_Y', ]
df = pd.read_csv('New.csv')
df = pd.DataFrame(df)
print(df)
#df.drop(['Image_Index', 'Finding_Labels'], axis=1, inplace=True)
df = df[['Image_Index', 'Pneumonia_Target']]
print(df, [...])
df2 = df[['Pneumonia_Target']]
print(df2)


filtered_data = df2["Pneumonia_Target"]==[1.0]
print('Filtered Data : ', filtered_data)




pnum = []
for Image_Index, Pneumonia_Target in df2.iteritems():
    if df2['Pneumonia_Target'] == [1.0]:
        print(Pneumonia_Target)
        pnum.append(Image_Index)

print('pnum Values:     ', pnum)
print('Completed SCraping')




PATH = '/Users/ryanjoseph/Desktop/Yale/Repositories/Pneumonia/Sample_ChestXRay_2GB/sample/sample/images/'

labels = df
for p in labels.itertuples():
    filepath = f'{PATH}/{p.Image_Index}'
    trainpath = f'{PATH}/{p.Pneumonia_Target}/{p.Image_Index}'

    os.rename(f'{filepath}', f'{trainpath}')



############################################ Data Augmentation ############################################

aug_start = time.time()

# Keras Image Preprocessing function
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

DIRECTORY = '/Users/ryanjoseph/Desktop/Yale/Repositories/Pneumonia/Sample_ChestXRay_2GB/sample/sample/images/'
#aug_set = datagen.flow_from_dataframe(labels_df,directory=DIRECTORY, x_col='', y_col='', target_size=(256, 256), color_mode='rgb')



aug_finish = (time.time() - aug_start) / 60
print(f'\Data Augmentation took {aug_finish} seconds!\n')

############################################ Learning Model ############################################

# CNN model
model = Sequential()


model.add(Conv2D(filters = 8, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = train_X.shape[1:]))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 3))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(len(disease_labels), activation = 'softmax'))

# compile model, run summary
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

















print('\n######################## Script Completed ########################\n')
#
