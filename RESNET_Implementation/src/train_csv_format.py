import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
from keras import models
from keras import layers
from keras import optimizers
import random 
import scipy.io
from keras.preprocessing.image import ImageDataGenerator
from random import randrange
import cv2
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import load_model
import os
import sys
from keras.callbacks import CSVLogger
import argparse
K.tensorflow_backend._get_available_gpus()


path  = os.path.abspath(os.path.join('..', 'models'))
sys.path.insert(0, path)

import  Resnet_multi_out 


#dat = pd.read_csv('../datasets/fgo_multiclass_labels.csv')

HEIGHT = 196
WIDTH = 196

dataset_path = "../datasets"
train_count= 0
test_count = 0
label = 0
width = 196
height = 196
CHANNELS = 3
dataset_location = dataset_path+"/RAP"+"/RAP_dataset"
train_datapath = dataset_path+'/'+'RAP_train_data.h5'
test_datapath = dataset_path+'/'+'RAP_test_data.h5'
datasets_path_annotation = dataset_path+'/RAP/'+'RAP_annotation/RAP_annotation.mat'

mat = scipy.io.loadmat(datasets_path_annotation)

def exctract_lables():
    columns = []
    data = mat["RAP_annotation"][0][0][3]
    for row in data: 
        label = row[0][0]
        columns.append(label)
    return columns

column = exctract_lables()
column.insert(0, "img_path")
dataset_df = pd.DataFrame(columns=column)

    
print(mat["RAP_annotation"][0][0][1][0].tolist())

# '# Length of mat["RAP_annotation"][0][0] = 7
# Index       length        attribute

# 0           5
# 1           41585           consist [ 0,1,1,,1,1,0.......1] '1' represent respective label exist and vide versa.
# 2           92              attribute label in chineses
# 3           92              attribute label name 
# 4           41585
# 5           41585           image file name
# 6           51    '
# 



EPOCHS = 50
BS = 8

def image_generator_fgo(king_of_lists, bs, mode="train", aug=None):
    # loop indefinitely
    
    while True:
        # initialize our batches of images and labels
        test_metrics = {'age': 'accuracy','obesity': 'accuracy',
               'hair': 'accuracy','glasses': 'accuracy',
               'hat': 'accuracy', 'upper_body':'accuracy', 'lower_body':'accuracy', 
               'attach_something':'accuracy', 'action':'accuracy', 'gender':'accuracy'}
        images = []
        #labels = []
        
        ages = []
        obesities = []
        hairs = []
        glass = []
        hats = []
        upper_bodies = []
        lower_bodies = []
        attach_things = []
        actions = []
        genders = []
        
        # keep looping until we reach our batch size
        while len(images) < bs:
            combined_label_list = []
            random_index = randrange(len(king_of_lists[0]))
            img = image.load_img(king_of_lists[0][random_index],target_size=(196, 196)) #read in image
            img = image.img_to_array(img)
            #img = cv2.resize(img, (224, 224))
            #F this making my own augmentations
            #rand = random.randint(1,101)
            #if rand < 50: 
            #    img = cv2.flip( img, 0 )# horizantal flip
            #rand = random.randint(1,101)
            #img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            
            #create labels
          
            obesities.append(np.array(king_of_lists[1][random_index])) # obesity
            genders.append(np.array(king_of_lists[2][random_index]))
             
            upper_bodies.append(np.array(king_of_lists[3][random_index]))
            
            attach_things.append(np.array(king_of_lists[4][random_index]))
            images.append(img)
            #labels.append(gender
            
        #labels = {'gender': np.array(gender), 'region': np.array(region),
        #        'fighting_style': np.array(fight),
        #         'alignment': np.array(alignment),'color': np.array(color)}
        labels = [np.array(obesities), np.array(genders),
                 np.array(upper_bodies),  np.array(attach_things)]
        #labels = [gender,region,fight,alignment,color]
        # if the data augmentation object is not None, apply it
        #labels
        if aug is not None:
            (images, labels) = next(aug.flow(np.array(images),labels, batch_size=bs))
        
        #print(labels.shape)
        # yield the batch to the calling function
        
        yield np.array(images),  labels 




aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")


dat = pd.read_csv("../datasets/RAP/data.csv")



def create_new_labels(row):
    if row["Female"]==1:
        row["Male"] = 0
        row["GenderNone"] = 0 
    elif row["Female"]>1:
        row["GenderNone"] = 1 
        row["Female"] = 0
        row["Male"] = 0
    else:
        row["Male"] = 1
        row["Female"] = 0
        row["GenderNone"] = 0
        
    if row["hs-Glasses"]==1:
        row["no-Glasses"] = 0
    else:
        row["no-Glasses"] = 1
        row["hs-Glasses"] = 0
    if row["hs-Hat"] == 1:
        row["no-Hat"] = 0 
    else:
        row["hs-Hat"] = 0 
        row["no-Hat"] = 1
        
    return row

dat = dat.apply (lambda row: create_new_labels(row), axis=1)


X = dat['img_path']
y = dat.loc[:, ~dat.columns.isin(['img_path', 'Unnamed: 0'])]
y = dat[[ 'BodyFat','BodyNormal', 'BodyThin',
       'ub-Shirt', 'ub-Sweater', 'ub-Vest', 'ub-TShirt', 'ub-Cotton',
       'ub-Jacket', 'ub-SuitUp', 'ub-Tight', 'ub-ShortSleeve', 'ub-Jacket', 'ub-SuitUp', 'ub-Tight', 'ub-ShortSleeve',
     	'attach-Backpack','attach-SingleShoulderBag', 'attach-HandBag', 'attach-Box',
       'attach-PlasticBag', 'attach-PaperBag', 'attach-HandTrunk','attach-Other',
      
      'Male', 'Female', 'GenderNone'
        ]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

#train


obesity = ['BodyFat',
       'BodyNormal', 'BodyThin']
obesity_train = y_train[obesity]
obesity_nodes = obesity_train.shape[1]
obesity_train = obesity_train.values.tolist()



upper_body = ['ub-Shirt', 'ub-Sweater', 'ub-Vest', 'ub-TShirt', 'ub-Cotton',
       'ub-Jacket', 'ub-SuitUp', 'ub-Tight', 'ub-ShortSleeve']

upper_body_train = y_train[upper_body]
upper_body_nodes = upper_body_train.shape[1]
upper_body_train = upper_body_train.values.tolist()


attach_something= ['attach-Backpack',
       'attach-SingleShoulderBag', 'attach-HandBag', 'attach-Box',
       'attach-PlasticBag', 'attach-PaperBag', 'attach-HandTrunk',
       'attach-Other']
attach_something_train = y_train[attach_something]
attach_something_nodes = attach_something_train.shape[1]
attach_something_train = attach_something_train.values.tolist()





gender = ['Male', 'Female', 'GenderNone']
gender_train = y_train[gender]
gender_nodes = gender_train.shape[1]
gender_train = gender_train.values.tolist()




obesity_test = y_test[obesity]
obesity_nodes = obesity_test.shape[1]
obesity_test = obesity_test.values.tolist()



upper_body_test = y_test[upper_body]
upper_body_nodes = upper_body_test.shape[1]
upper_body_test = upper_body_test.values.tolist()



attach_something_test = y_test[attach_something]
attach_something_nodes = attach_something_test.shape[1]
attach_something_test = attach_something_test.values.tolist() 



gender_test = y_test[gender]
gender_nodes = gender_test.shape[1]
gender_test = gender_test.values.tolist() 


print("obesity train",obesity_train)



train_lists = [X_train, obesity_train, gender_train,  upper_body_train
              , attach_something_train]
test_lists = [X_test,obesity_test, gender_test, upper_body_test, attach_something_test]
#train_lists = [X_train, region_train, fighting_style_train, alignment_train, color_train]
#test_lists = [X_test, region_test, fighting_style_test, alignment_test, color_test]
# initialize both the training and testing image generators
trainGen = image_generator_fgo(train_lists, BS, 
                               mode="train", aug=None)
testGen = image_generator_fgo(test_lists, BS, 
                              mode="train", aug=None)


losses = {"Body_Output": "categorical_crossentropy",
              "Gender_Output": "categorical_crossentropy",
              "Dressing_Output": "categorical_crossentropy",
              "Bag_Output": "categorical_crossentropy" }

lossWeights = {"Body_Output": 1.0, "Gender_Output": 1.0, "Dressing_Output": 1.0, "Bag_Output": 1.0}

dropout = 0.5
dd = 0.0





print("Number Of Nodes", obesity_nodes, gender_nodes, upper_body_nodes, attach_something_nodes)

def  main(args):
    filepath="bin/" + "Multinet" + "_model_weights.h5"
    checkpoint = ModelCheckpoint(filepath='bin/resnet50_{epoch:08d}.h5', verbose=1, save_weights_only = False,save_best_only=False, mode='max')
    csv_logger = CSVLogger('log/log.csv', append=False, separator=';')
    callbacks_list = [checkpoint,csv_logger]
    adam = Adam(lr=0.00001)  

    if args.model != None:
        multi_model = load_model(args.model)
        multi_model.compile(adam, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])

        multi_model.fit_generator(trainGen,steps_per_epoch=len(X_train) // BS,
                    validation_data=testGen,
                    validation_steps=len(X_test) // BS,
                    epochs=EPOCHS,callbacks=callbacks_list)


    else:

        number_category = [obesity_nodes, gender_nodes, upper_body_nodes, attach_something_nodes]

        multi_model = Resnet_multi_out.ResNet50(weight_path=dataset_path+'/imagenet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                                    include_top=False, 
                                    input_shape=(HEIGHT, WIDTH, 3),num_dense = number_category)


        multi_model.compile(adam, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])


        multi_model.fit_generator(trainGen,steps_per_epoch=len(X_train) // BS,
                    validation_data=testGen,
                    validation_steps=len(X_test) // BS,
                    epochs=EPOCHS,callbacks=callbacks_list)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    parser.add_argument('--model', help='Model to use.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

#Command 
'''
sudo python3 transfer_image_batch.py --model bin/resnet50_00000007.h5 
'''

