#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 11:28:02 2019

@author: jayasoo
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

train_df = pd.read_csv("training.csv")
train_df.dropna(inplace=True)

def getImage(image):
    image = image.split(" ")
    image = list(map(int, image))
    image = np.reshape(image, (96,96))
    return image

def flipImage(X, Y):
    """Function to flip images and their corresponding keypoints"""
    XFlip = X[:,:,::-1]
    
    symmetric_flip_indices = [
    (0, 2), (1, 3),
    (4, 8), (5, 9), (6, 10), (7, 11),
    (12, 16), (13, 17), (14, 18), (15, 19),
    (22, 24), (23, 25),
    ]
    
    asymmetric_flip_indices = [(20,21), (26,27), (28,29)]
    
    YFlip = np.zeros((Y.shape[0], 30))
    for i,j in symmetric_flip_indices[::2]:
        YFlip[:,j] = 96 - Y[:,i]
        YFlip[:,i] = 96 - Y[:,j]
    for i,j in symmetric_flip_indices[1::2]:
        YFlip[:,j] = Y[:,i]
        YFlip[:,i] = Y[:,j]
    
    
    for i,j in asymmetric_flip_indices:
        YFlip[:,i] = 96 - Y[:,i]
        YFlip[:,j] = Y[:,j]
    
    return XFlip, YFlip

def getModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', input_shape=(96,96,1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(32, kernel_size=(5,5), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Dropout(0.2),
                               
        tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(30, activation='relu')
    ])
    return model

if __name__ == '__main__':
    X_train = train_df['Image'].apply(getImage)
    X_train = np.stack(x for x in X_train)
    
    Y_train = train_df[train_df.columns[:-1]].values
    
    X_flipped, Y_flipped = flipImage(X_train, Y_train)
    X_train = np.append(X_train, X_flipped, axis=0)
    Y_train = np.append(Y_train, Y_flipped, axis=0)
    X_train = np.expand_dims(X_train,-1)
    X_train = X_train / 255.
    
    y_scaler = MinMaxScaler((-1,1))
    Y_train = y_scaler.fit_transform(Y_train)
    
    with open('y_scaler.pickle', 'wb') as handle:
        pickle.dump(y_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    model = getModel()
    model.compile(loss='mse', optimizer='adam')
    ckpt_file_path = "./weights.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(ckpt_file_path, save_best_only=False, monitor='val_acc', mode='max')
    model.fit(X_train, Y_train, epochs=100, shuffle=True, validation_split=0.2, callbacks=[checkpoint])