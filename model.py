# Loading necessary python libraries
import csv
import cv2
import numpy as np
import tensorflow
import pickle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Activation
from keras.layers import Lambda, Cropping2D, Dropout, ELU
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

###-----------------------------------------------------------------------
# Loading dataset 
###-----------------------------------------------------------------------
# Loading CSV to get URL to images and steering angles
def load_driving_log(dataset_path):
    lines = []
    with open(dataset_path+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the headers
        for line in reader:
            lines.append(line)
    return(lines)

# Loading images from URLs and storing measurements
def load_dataset(dataset_path, lines):
    images = []
    measurements = []
    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = dataset_path + 'IMG/' + filename
            #img = mpimg.imread(current_path)
            img = cv2.imread(current_path)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(rgb_image)
            
            steering_angle = float(line[3]) # steering angle for img in iteration
            if i == 0:
                measurements.append(steering_angle)
            elif i == 1:
                measurements.append(steering_angle + 0.20)
            else:
                measurements.append(steering_angle - 0.20)
        
    return(images, measurements)

# Loading augmented images and respective measurements
def load_aug_dataset(images, measurements):
    aug_images = []
    aug_measurements = []
    for img, steering_angle in zip(images, measurements):
        aug_images.append(img)
        aug_measurements.append(steering_angle)
        aug_images.append(cv2.flip(img,1))
        aug_measurements.append(steering_angle*-1.0)
        
    return(aug_images, aug_measurements)

dataset_path = '/home/workspace/CarND-Behavioral-Cloning-P3/data/'
#lines = load_driving_log(dataset_path)
#images, measurements = load_dataset(dataset_path, lines)
#aug_images, aug_measurements = load_aug_dataset(images, measurements)

# Writing above datasets to pickle objects
results_path = '/opt/results/'
#pickle.dump(images, open( results_path+"images.p", "wb" ) )
#pickle.dump(measurements, open( results_path+"measurements.p", "wb" ) )

# Loading datasets from pickle files
#lines = pickle.load( open( results_path+"lines.p", "rb" ) )
images = pickle.load( open( results_path+"images.p", "rb" ) )
measurements = pickle.load( open( results_path+"measurements.p", "rb" ) )
aug_images, aug_measurements = load_aug_dataset(images, measurements)

# Getting X and y datasets
#X_train = np.array(images)
#y_train = np.array(measurements)
X_train_aug = np.array(aug_images)
y_train_aug = np.array(aug_measurements)

print(X_train_aug.shape)     # output: (24108, 160, 320, 3)
print(X_train_aug.shape[0])  # output: 24108
print(X_train_aug.shape[1:]) # output: (160, 320, 3)

###-----------------------------------------------------------------------
# # Training CNN models
###-----------------------------------------------------------------------
# LeNet-5 Architecture
'''
model = keras.Sequential()
model.add(Lambda(lambda x: (x/127.5) - 1.0, input_shape=(160, 320, 3))) # pre-processing
model.add(Cropping2D(cropping=((75, 25), (0, 0)))) # cropping images
model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
model.fit(X_train, y_train, validation_split=0.2,
          shuffle=True, epochs=5)
model.save('model.h5')
'''
### Nvidia Architecture
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) # pre-processing
model.add(Cropping2D(cropping=((75, 25), (0, 0)))) # cropping images
# 5 Convoluation layers as in Nvidia's papers
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))
model.add(Conv2D(64, (3, 3), activation="elu"))
#model.add(Conv2D(64, (3, 3), activation="elu"))
model.add(Dropout(0.5)) # adding dropout
model.add(Flatten()) # Flatten layer
# Layer 1
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dropout(0.25)) # adding dropout
# Layer 2
model.add(Dense(50))
model.add(Activation('elu'))
# Layer 3
model.add(Dense(10))
model.add(Activation('elu'))
model.add(Dropout(0.25)) # adding dropout
# # Layer 4: Final output layer
model.add(Dense(1))

# Displaying summary of the model
model.summary() 

# Compile and train the above network
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_aug, y_train_aug, validation_split=0.2,
          batch_size=32, epochs=5, shuffle=True)

# Saving the model to a file
model.save('model_nvidia_aug.h5')
print("Model has been trained & saved!")



