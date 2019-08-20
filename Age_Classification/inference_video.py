import cv2
import numpy as np 
import tensorflow as tf 
from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse

from keras.applications.resnet50 import ResNet50, preprocess_input
import cv2
import numpy as np 
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
import os
from keras.preprocessing import image
from utils import crop_and_save as crop_and_save

from PIL import Image

HEIGHT = 212
WIDTH = 212
base_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape=(HEIGHT, WIDTH, 3))

dropout = 0.5
FC_LAYERS = [512, 256] 
adam = Adam(lr=0.00001)  

detection_graph, sess = detector_utils.load_inference_graph()



def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x) 
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model





def predict(model ,imgs):

    
    class_list = ["ONE","TWO","PALM","PUNCH","THUMBS_DOWN","THUMBS_UP","CLICK"]









    cv_img = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    cv_img =  Image.fromarray(cv_img, 'RGB')
    cv_img = cv_img.resize((212,212))
    cv_img = image.img_to_array(cv_img)

    y = np.expand_dims(cv_img, axis=0)

    y = preprocess_input(y)




    prediction_score = model.predict(y)

    data = prediction_score[0]

    
    data= data.tolist()
    print(data.index(max(data)) )

    print(class_list[data.index(max(data))])
    #     # y = 20







weight = "bin/Hand_Classification-0030.ckpt"

class_list = ["ONE","TWO","PALM","PUNCH","THUMBS_DOWN","THUMBS_UP","CLICK"]

model = build_finetune_model(base_model, dropout=dropout, fc_layers=FC_LAYERS, num_classes=len(class_list))
model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights(weight)


num_hands_detect = 2
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
score_thresh = 0.2
num_frames = 0
start_time = datetime.datetime.now()



    
while True:

    ret, image_np = cap.read()

    if ret :


        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")


        orig_img = image_np.copy()

        im_height, im_width  = image_np.shape[:2]   
        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

        # draw bounding boxes on frame
        detector_utils.draw_box_on_image(num_hands_detect, score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np)

        output_path = "hello_dummpy"
        file = "hello_dummpy"

        hand = crop_and_save.crop_hand(num_hands_detect, score_thresh,scores, boxes,orig_img,output_path ,file)

        

        # hand= detector_utils.crop_hand(num_hands_detect, score_thresh,
        #                                  scores, boxes, im_width, im_height,
        #                                  image_np)


        try :
            hand = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)

            predict(model,hand)
        except Exception as e:

            print (getattr(e, 'message', repr(e)))


        # Calculate Frames per second (FPS)





        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

  
        detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)


        cv2.imshow('Single-Threaded Detection',cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        try:
           
            cv2.imshow("cropped",hand)
        except:
            pass

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))
