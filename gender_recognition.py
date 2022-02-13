#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python script to train the AI able to recognize the gender of someone in a picture

@author: Deronsart Nicolas
"""

import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from keras.preprocessing import image



def data_preprocessing(directory):
    '''
        Function which prepares the images in two parts:
            - The images to train the AI -> training_set
            - The one to test its efficiency -> test_set 
            
        Parameters :
            The name of the file containing the pictures
            
        Return:
            training_set and test_set
    '''
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, 
                                       zoom_range=0.2, horizontal_flip=True)
    
    training_set = train_datagen.flow_from_directory(directory+'/training_set',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory(directory+'/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
    
    return training_set, test_set


def build_cnn():
    '''
        Function to build the CNN
    
        Return :
            The CNN (Convolutional Neural Network)
    '''
    cnn = tf.keras.models.Sequential()
    
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, 
                                   activation='relu', input_shape=[64, 64, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, 
                                   activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    cnn.add(tf.keras.layers.Flatten())
    
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    return cnn


def predict_woman_man(file, cnn):
    '''
        Function to give the prediction of the AI for a picture given
    
        Return :
            1 for a woman
            0 for a man
    '''
    test_image = image.load_img(file, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
    result = cnn.predict(test_image)
    
    return result[0][0]
    


if __name__ == "__main__" :
    
    directory = "dataset_women_men"
    
    cnn = build_cnn()
        
    if os.path.isdir('checkpoints'):
        print("=> Loading checkpoint...")
        cnn.load_weights('./checkpoints/checkpoint_women_men')
        print("Done !\n")
        
    else:
        training_set, test_set = data_preprocessing(directory)
        
        cnn.compile(optimizer='adam', loss='binary_crossentropy', 
                    metrics=['accuracy'])
        cnn.fit(training_set, validation_data=test_set, epochs=15)
        
        cnn.save_weights('./checkpoints/checkpoint_women_men')
    
    # Test
    test_file = directory + "/single_prediction/man_or_woman_"
    print("Test Man 1 : "+str(predict_woman_man(test_file+"1.jpg", cnn)))
    print("Test Woman 1 : "+str(predict_woman_man(test_file+"2.jpg", cnn)))
    print("Test Man 2 : "+str(predict_woman_man(test_file+"3.jpg", cnn)))
    print("Test Woman 2 : "+str(predict_woman_man(test_file+"4.jpg", cnn)))


