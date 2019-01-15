# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 16:35:13 2018

@author: crist
@dataset: https://www.kaggle.com/iarunava/happy-house-dataset
"""

import h5py
import numpy as np
TRAINING_DATA_PATH = "dataset/train_happy.h5"
TEST_DATA_PATH = "dataset/test_happy.h5"

def load_dataset():
    train_data = h5py.File(TRAINING_DATA_PATH, "r")
    x_train = np.array(train_data["train_set_x"][:]) 
    y_train = np.array(train_data["train_set_y"][:]) 

    test_data = h5py.File(TEST_DATA_PATH, "r")
    x_test = np.array(test_data["test_set_x"][:])
    y_test = np.array(test_data["test_set_y"][:]) 
    
    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))
    
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

X_train, y_train, X_test, y_test = load_dataset()

#data scalling
X_train = X_train/255.
X_test = X_test/255.
y_train = y_train.T
y_test = y_test.T

# kaggle best solution
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 input_shape = (64,64,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))


#Output Layer
model.add(Dense(units = 1,kernel_initializer="uniform", activation = 'sigmoid'))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compiling Neural Network
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size=20, epochs=35)
scores = model.evaluate(X_test, y_test)


# MY SOLUTION

from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout
from keras.models import Sequential

model = Sequential()
#Conv layer 1
model.add(Conv2D(filters=20, kernel_size = (3, 3), padding = "same", strides=(1,2), input_shape=(64, 64, 3), kernel_initializer='uniform',activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

#Conv layer 2
model.add(Conv2D(filters=20, kernel_size = (3, 3), padding = "same", strides=(1,2), kernel_initializer='uniform',activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

#Flatten layer
model.add(Flatten())

#Fully connected layers
model.add(Dense(units = 64, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 64, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.2))

#output layer
model.add(Dense(units = 1, activation='sigmoid'))

model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=10, epochs=50)
scores = model.evaluate(X_test, y_test)
