import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers

dataset_path = '/home/workspace/CarND-Behavioral-Cloning-P3/data/'
samples = []
with open(dataset_path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

def generator(samples, batch_size, dataset_path):
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        num_samples = len(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = dataset_path + 'IMG/' + filename
                    img = mpimg.imread(current_path)
                    images.append(images)

                    steering_angle = float(batch_sample[3]) # steering angle for img in iteration
                    if i == 0:
                        measurements.append(steering_angle)
                    elif i == 1:
                        measurements.append(steering_angle + 0.20)
                    else:
                        measurements.append(steering_angle - 0.20)
            '''        
            aug_images = []
            aug_measurements = []
            for img, steering_angle in zip(images, measurements):
                aug_images.append(img)
                aug_measurements.append(steering_angle)
                aug_images.append(cv2.flip(img,1))
                aug_measurements.append(steering_angle*-1.0)
                
            X_train = np.array(aug_images)
            y_train = np.array(aug_measurements)
            '''
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(samples=train_samples, batch_size=batch_size, dataset_path=dataset_path)
validation_generator = generator(samples=validation_samples, batch_size=batch_size, dataset_path=dataset_path)

# Convert images to gray scale
def converter(im):
    import tensorflow as tf
    return tf.image.rgb_to_grayscale(im)

####### I. Create model
model = Sequential()
### 1 CNN part
model.add(Lambda(lambda x:x/127.5-1,input_shape=(160,320,3)))
model.add(Lambda(converter,output_shape=(160,320,1)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(filters=16,kernel_size=(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=32,kernel_size=(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
model.add(Dropout(0.5))
### 2 FC part
model.add(Flatten()) #L1 flatten input
model.add(Dense(128)) #L2 FC -> Relu
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64)) #L2 FC -> Relu #L3 FC -> Relu
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1)) #L3 FC for regression

model.summary()

####### II train & save model
model.compile(optimizer='adam',loss='mse')
model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=6*len(train_samples),
                    validation_steps=6*len(validation_samples), 
                    epochs=7,verbose=1,workers=8,use_multiprocessing=True)
model.save('model_generator.h5')