#coding=utf-8
import keras
import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Embedding
from keras.optimizers import Adam, SGD
from keras.layers import LSTM
from sklearn.ensemble import RandomForestClassifier

def get_text_model(input_shape,output):
    top_words, max_words = input_shape
    model = Sequential()
    model.add(Embedding(top_words, 128, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250))
    model.add(Activation('relu'))
    model.add(Dense(output))
    return model


def get_credit_model():
   # model = RandomForestClassifier(n_estimators = 50, random_state=10)
    
    model = Sequential()
    model.add(Dense(30, input_dim=30, activation='relu')) 
    model.add(Dense(5, activation='relu'))     # kernel_initializer='normal'
    model.add(Dense(2))                 # kernel_initializer='normal'
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_image_model(in_shape, output):

    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='Same', input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), padding='Same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(output))
    return model


