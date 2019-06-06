
# Author : Shunt-OS ,16-May-2019


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import cv2
import re
import h5py
import numpy as np

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

path  = os.path.abspath(os.path.join('..', 'models'))

sys.path.insert(0, path)
import resnet50
import utils

def extract_number(input):
    few_input = input[:10]

    numbers = re.split('(\d+)',few_input)
    return numbers[1]


def main(args):

    classes = 6
    input_shape = (64,64,3)

    train(classes,input_shape,args,sess,epoch,image_list,label_list)
    return 1


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def load_dataset():
    dataset_path  = os.path.abspath(os.path.join('..', 'datasets'))
    train_dataset = h5py.File(dataset_path+'/'+'train_data.h5', "r")
    train_set_x_orig = np.array(train_dataset["Train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["Train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(dataset_path+'/'+'test_data.h5', "r")
    test_set_x_orig = np.array(test_dataset["Train_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["Train_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def train_model():

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 2).T
    Y_test = convert_to_one_hot(Y_test_orig, 2).T

    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
          
    


    model = resnet50.ResNet50(input_shape = (64, 64, 3), classes = 2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs = 5, batch_size = 32)

    model.save_weights("final.h5")

                            
def prediction():
    model =     model = resnet50.ResNet50(input_shape = (64, 64, 3), classes = 2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    img_path = 'imgs/peta.bmp'
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print(model.predict(x))

def train(classes, input_shape, args, sess, epoch, image_list, label_list ):

    datagen_train = ImageDataGenerator()
    datagen_validation = ImageDataGenerator()
    model = resnet50.ResNet50()
    resnet50.model.fit(X_train, Y_train, epochs = 25, batch_size = 32)
    model.save('bin/weight.h5')

train_model()


