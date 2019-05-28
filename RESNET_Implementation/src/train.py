
# Author : Shunt-OS ,16-May-2019
#Ekbana

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

# sys.path.insert(0, '/home/ekbana/oddocker/person_reid/custom_reid/models')
sys.path.insert(0, path)
import resnet50
import utils


def extract_number(input):
    few_input = input[:10]

    numbers = re.split('(\d+)',few_input)
    return numbers[1]

    # numbers = re.findall('\d+',few_input)
    # numbers = map(int,numbers)
    # number = (max(numbers))

    # return str(number)




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






def parse_arguments(argv):

    parser = argparse.ArgumentParser() 

    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization', 
        help='Performs fixed standardization of images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
        help='Loss based on the norm of the activations in the prelogits layer.', default=0.0)
    parser.add_argument('--prelogits_norm_p', type=float,
        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--prelogits_hist_max', type=float,
        help='The max value for the prelogits histogram.', default=10.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
    parser.add_argument('--validate_every_n_epochs', type=int,
        help='Number of epoch between validation', default=5)
    parser.add_argument('--validation_set_split_ratio', type=float,
        help='The ratio of the total dataset to use for validation', default=0.0)
    parser.add_argument('--min_nrof_val_images_per_class', type=float,
        help='Classes with fewer images will be removed from the validation set', default=0)




# if __name__ == '__main__':
#     main(parse_arguments(sys.argv[1:]))


#train_model()

prediction()


