import cv2
import datetime as dt
import h5py
import numpy as np
import os
from glob import glob
import re
import random 


def extract_number(input):
    few_input = input[:10]

    numbers = re.split('(\d+)',few_input)
    return numbers[1]

def create_h5_dataset(classes,data):

    asciiList = [n.encode("ascii", "ignore") for n in classes]
    data.create_dataset('list_classes', (len(asciiList),1),'S10', asciiList)

    data.create_dataset("Train_set_x", (20000,64,64,3),maxshape = (None,64,64,3),dtype= 'uint8',chunks = True)
    data.create_dataset("Train_set_y", (20000,),maxshape = (None,),dtype='i8',chunks = True)
    return data


def get_images():
    """ 
    Saves Compressed , resized images as HDF5 datasets

    Returns:
     data.h5, where each dataset is an image of class label

     x1,y1 =    image and corresponding class label

    """

    width = 64
    height = 64
    CHANNELS = 3
    label = 0
    train_count = 0
    test_count = 0
    classes = ["men","women"]
    data = "PETA"

    dataset_path  = os.path.abspath(os.path.join('..', 'datasets'))

    
    if data == "PETA":
        dataset_location = dataset_path+"/"+"/PETA dataset"
        train_datapath = dataset_path+'/'+'train_data.h5'
        test_datapath = dataset_path+'/'+'test_data.h5'

        with h5py.File(train_datapath, 'w') as hf1: 
            with h5py.File(test_datapath,'w') as hf2:

  
                hf1 =create_h5_dataset(classes, hf1)
                hf2= create_h5_dataset(classes, hf2)

                i = 0
                for data_folder in os.listdir(dataset_location):
                    image_path = dataset_location+'/'+data_folder+'/archive'

                    label_path = image_path+'/'+'Label.txt'
       
                    with open(label_path, 'r') as fh:
                        for line in fh:
                            img_id = extract_number(line)

                            for file in os.listdir(image_path):
   
                                filename, file_extension = os.path.splitext(file)                          
                                file_string = (filename.split("_"))[0]

                                if(file_string == img_id):
                   

                                    if file != "Label.txt":
                                        image_location = image_path+'/'+file
                                        img = cv2.resize(cv2.imread(image_location, cv2.IMREAD_COLOR),(width,height), interpolation = cv2.INTER_CUBIC)


                                        r = random.randint(0,1000)
                                        if  "personalFemale"  in line: 
                                            label = 0
                                        elif "personalMale" in line:
                                            label = 1
                                        

                                        else:
                                            print("Error")
                                            print(file)

                                        if r >100:
                                            hf1["Train_set_x"][train_count,...] = img
                                         
                                            hf1["Train_set_y"][train_count] = label
                                            train_count +=1

                                        else:

                                            hf2["Train_set_y"][test_count] = label
                                            hf2["Train_set_x"][test_count, ...] = img
                                            test_count +=1
                                    
                                        i+=1

                print(train_count)
                print(test_count)
                tot = train_count +test_count

                print(tot)

                hf2["Train_set_x"].resize((test_count,64,64,3))
                hf2["Train_set_y"].resize((test_count,))
                print(hf2["Train_set_x"])
                print(hf2["Train_set_y"])

            hf1["Train_set_x"].resize((train_count,64,64,3))
            hf1["Train_set_y"].resize((train_count,))
            print(hf1["Train_set_x"])
            print(hf1["Train_set_y"])

def test_data():
    dataset_path  = os.path.abspath(os.path.join('..', 'datasets'))
    with h5py.File(dataset_path+'/'+'train_data.h5', "r") as hf:

        #for i in range(0,1000):
        print(hf.keys())
        print(hf["Train_set_x"])
        print(hf["Train_set_y"])


        img = hf["Train_set_x"][181]

        cv2.imwrite("img1.jpg",img)
        print(hf["Train_set_y"][181])

        for i in range(len(hf["Train_set_y"])):
            img = hf["Train_set_x"][i]
            label = str(hf["Train_set_y"][i])

            name = "imgs/img"+str(i)+"_"+label+".jpg"

            cv2.imwrite(name,img)
           

        

#

#get_images()
test_data()

