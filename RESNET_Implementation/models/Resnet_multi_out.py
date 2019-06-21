"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
   
    bn_axis = 3
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(
            weight_path,
            include_top,
             input_shape,
             num_dense,
             
             ):


    FC_LAYERS = [1024, 1024]
    img_input = layers.Input(shape=input_shape)
    
    bn_axis =3

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')


########################  SUB-BLOCK 1 ################################

    x_body = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x_body = identity_block(x_body, 3, [128, 128, 512], stage=3, block='b')
    x_body = identity_block(x_body, 3, [128, 128, 512], stage=3, block='c')
    x_body = identity_block(x_body, 3, [128, 128, 512], stage=3, block='d')

    x_body = conv_block(x_body, 3, [256, 256, 1024], stage=4, block='e')
    x_body = identity_block(x_body, 3, [256, 256, 1024], stage=4, block='f')
    x_body = identity_block(x_body, 3, [256, 256, 1024], stage=4, block='g')
    x_body = identity_block(x_body, 3, [256, 256, 1024], stage=4, block='h')
    x_body = identity_block(x_body, 3, [256, 256, 1024], stage=4, block='i')
    x_body = identity_block(x_body, 3, [256, 256, 1024], stage=4, block='j')

    x_body = conv_block(x_body, 3, [512, 512, 2048], stage=5, block='k')
    x_body = identity_block(x_body, 3, [512, 512, 2048], stage=5, block='l')
    x_body = identity_block(x_body, 3, [512, 512, 2048], stage=5, block='m')
    x_body = Flatten()(x_body)

    
######################### SUB-BLOCK 2 #######################################


    x_gender = conv_block(x, 3, [128, 128, 512], stage=6, block='n')
    x_gender = identity_block(x_gender, 3, [128, 128, 512], stage=6, block='o')
    x_gender = identity_block(x_gender, 3, [128, 128, 512], stage=6, block='p')
    x_gender = identity_block(x_gender, 3, [128, 128, 512], stage=6, block='q')

    x_gender = conv_block(x_gender, 3, [256, 256, 1024], stage=7, block='r')
    x_gender = identity_block(x_gender, 3, [256, 256, 1024], stage=7, block='s')
    x_gender = identity_block(x_gender, 3, [256, 256, 1024], stage=7, block='t')
    x_gender = identity_block(x_gender, 3, [256, 256, 1024], stage=7, block='u')
    x_gender = identity_block(x_gender, 3, [256, 256, 1024], stage=7, block='v')
    x_gender = identity_block(x_gender, 3, [256, 256, 1024], stage=7, block='w')

    x_gender = conv_block(x_gender, 3, [512, 512, 2048], stage=8, block='x')
    x_gender = identity_block(x_gender, 3, [512, 512, 2048], stage=8, block='y')
    x_gender = identity_block(x_gender, 3, [512, 512, 2048], stage=8, block='z')
    x_gender = Flatten()(x_gender)


########################## SUB-BLOCK 3 #########################################

    x_dressing = conv_block(x, 3, [128, 128, 512], stage=9, block='aa')
    x_dressing = identity_block(x_dressing, 3, [128, 128, 512], stage=3, block='bb')
    x_dressing = identity_block(x_dressing, 3, [128, 128, 512], stage=3, block='cc')
    x_dressing = identity_block(x_dressing, 3, [128, 128, 512], stage=3, block='dd')

    x_dressing = conv_block(x_dressing, 3, [256, 256, 1024], stage=10, block='ee')
    x_dressing = identity_block(x_dressing, 3, [256, 256, 1024], stage=10, block='ff')
    x_dressing = identity_block(x_dressing, 3, [256, 256, 1024], stage=10, block='gg')
    x_dressing = identity_block(x_dressing, 3, [256, 256, 1024], stage=10, block='hh')
    x_dressing = identity_block(x_dressing, 3, [256, 256, 1024], stage=10, block='ii')
    x_dressing = identity_block(x_dressing, 3, [256, 256, 1024], stage=10, block='jj')

    x_dressing = conv_block(x_dressing, 3, [512, 512, 2048], stage=11, block='kk')
    x_dressing = identity_block(x_dressing, 3, [512, 512, 2048], stage=11, block='ll')
    x_dressing = identity_block(x_dressing, 3, [512, 512, 2048], stage=11, block='mm')
    x_dressing = Flatten()(x_dressing)


########################## SUB-BLOCK 3 #########################################

    x_bag = conv_block(x, 3, [128, 128, 512], stage=12, block='nn')
    x_bag = identity_block(x_bag, 3, [128, 128, 512], stage=12, block='oo')
    x_bag = identity_block(x_bag, 3, [128, 128, 512], stage=12, block='pp')
    x_bag = identity_block(x_bag, 3, [128, 128, 512], stage=12, block='qq')

    x_bag = conv_block(x_bag, 3, [256, 256, 1024], stage=13, block='rr')
    x_bag = identity_block(x_bag, 3, [256, 256, 1024], stage=13, block='ss')
    x_bag = identity_block(x_bag, 3, [256, 256, 1024], stage=13, block='tt')
    x_bag = identity_block(x_bag, 3, [256, 256, 1024], stage=13, block='uu')
    x_bag = identity_block(x_bag, 3, [256, 256, 1024], stage=13, block='vv')
    x_bag = identity_block(x_bag, 3, [256, 256, 1024], stage=13, block='ww')

    x_bag = conv_block(x_bag, 3, [512, 512, 2048], stage=14, block='xx')
    x_bag = identity_block(x_bag, 3, [512, 512, 2048], stage=14, block='yy')
    x_bag = identity_block(x_bag, 3, [512, 512, 2048], stage=14, block='zz')
    x_bag = Flatten()(x_bag)


####################################  BLOCK END ###############################
    dropout = 0.5

    for fc in FC_LAYERS:
        x_body = Dense(fc, activation='relu')(x_body) 
        x_body = Dropout(dropout)(x_body)
  
        x_gender = Dense(fc, activation='relu')(x_gender) 
        x_gender = Dropout(dropout)(x_gender)

        x_dressing = Dense(fc, activation='relu')(x_dressing) 
        x_dressing = Dropout(dropout)(x_dressing)

        x_bag = Dense(fc, activation='relu')(x_bag) 
        x_bag = Dropout(dropout)(x_bag)    



    Branch_Body = Dense(num_dense[0], activation='softmax', name = "Body_Output")(x_body) 
    Branch_Gender = Dense(num_dense[1], activation='softmax', name = "Gender_Output")(x_gender) 
    Branch_Dressing = Dense(num_dense[2], activation='softmax', name = "Dressing_Output")(x_dressing) 
    Branch_Bag = Dense(num_dense[3], activation='softmax', name = "Bag_Output")(x_bag) 

    model = Model(img_input, outputs = [Branch_Body, Branch_Gender, Branch_Dressing, Branch_Bag], name='RootNet')

 
    #model.load_weights(weight_path)
       


    return model
