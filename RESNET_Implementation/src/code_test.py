import os
import numpy as np
import tensorflow as tf
import h5py
import math
import cv2

def load_dataset():
    train_dataset = h5py.File('data.h5', "r")

    data2 = h5py.File('train_signs.h5', "r")

    data3 = h5py


    #print(train_dataset.keys())

    print(data2.keys())

    print(data2["list_classes"])
    print(data2["train_set_x"])
    print(data2["train_set_y"])
    #train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    #train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    #print(type(train_set_x_orig.shape))



    #classes = np.array(train_dataset["list_classes"][:]) # the list of classes


    
    #train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))


#load_dataset()

def test_debug():
    with h5py.File("../src/train_data.h5",'w') as hf:
        print(hf.keys())

        #print(hf["Train_set_x"])

        #hf["Train_set_y"].resize((19000,))

        

    with h5py.File("train_signs.h5",'w') as hf2:  

        print(hf2["train_set_x"]) 




def test_create():
    with h5py.File('data.h5', 'w') as hf: 
       #g1 = hf.create_group("list_classes")
       #hf.create_group("train_set_x")
       #hf.create_group("train_set_y")
       hf.create_dataset("list_classes", (100,), dtype='i8')
       hf.create_dataset("Train_set_x", (10000,64,64,3),np.int8)

       hf.create_dataset("Train_set_y", (100,),dtype='i8')

       image_location = "imgs/a.png"

       img = cv2.resize(cv2.imread(image_location, cv2.IMREAD_COLOR),(width,height), interpolation = cv2.INTER_CUBIC)
       print("here")


def test():
    X = []
    with h5py.File('train_data.h5', 'w') as hf: 

        for file in os.listdir('imgs'):
            image_location = 'imgs'+'s/'+file
            img = cv2.resize(cv2.imread(image_location, cv2.IMREAD_COLOR),(60,60), interpolation = cv2.INTER_CUBIC)
            X.append(img)
        print(np.array(X).shape)
        hf.create_dataset(name='Train-X',
                                data=X,
                                shape=(10,60,60,3),
                                maxshape=(10,60,60,3),
                                compression="gzip",
                                compression_opts=9)
         


   
#load_dataset()
#test()

#test2()
#test_create()

test_debug()