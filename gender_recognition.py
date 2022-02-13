#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script python permettant l'entraînement d'une Intelligence Artificielle 
permettant de reconnaître le genre de la personne depuis une photo

@author: Deronsart Nicolas
"""

import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from keras.preprocessing import image



def data_preprocessing(directory):
    '''
        Fonction qui prépare les images en 2 ensembles:
            - Les images pour entraîner l'IA -> training_set
            - Celles pour tester son efficacité -> test_set 
            
        Param :
            Le nom du fichier dans lequel se trouvent les images
            
        Return:
            training_set et test_set
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
        Fonction qui construit l'Intelligence Artificielle 
    
        Return :
            Le CNN (Convolutional Neural Network)
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
        Fonction qui permet de donner la prediction du CNN pour une image 
        donnée
    
        Return :
            1 si c'est une femme
            0 si c'est un homme
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
    print("Man : "+str(predict_woman_man(test_file+"1.jpg", cnn)))
    print("Woman : "+str(predict_woman_man(test_file+"2.jpg", cnn)))
    print("Man : "+str(predict_woman_man(test_file+"3.jpg", cnn)))
    print("Woman : "+str(predict_woman_man(test_file+"4.jpg", cnn)))
    print("Man : "+str(predict_woman_man(test_file+"5.jpg", cnn)))
    print("Woman : "+str(predict_woman_man(test_file+"6.jpg", cnn)))


