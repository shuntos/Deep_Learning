''''
    Author:Shuntos 
    2020 Apr 14
'''



import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
import cv2
import time

start = time.time()
path_to_pb = "./out/froze_xxx.pb"
classes = ["class1","class2"]
dir_path = 'test_images'


try:

    if not os.path.exists(dir_path):
        print("rrNo such directory")
        raise Exception
    
    # Walk though all testing images one by one
    for root, dirs, files in os.walk(dir_path):
        for name in files:

            print("")
            image_path = name
            filename = dir_path +'/' +image_path
            print(filename)
            image_size=128
            num_channels=3
            images = []
        
            if os.path.exists(filename):
                # Reading the image using OpenCV
                image = cv2.imread(filename)
                # Resizing the image to our desired size and preprocessing will be done exactly as done during training
                image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
                images.append(image)
                images = np.array(images, dtype=np.uint8)
                images = images.astype('float32')
                images = np.multiply(images, 1.0/255.0) 
            
                # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
                x_batch = images.reshape(1, image_size,image_size,num_channels)

                # Let us restore the saved model 

                with tf.Session() as sess:
                   print("load graph")
                   with gfile.FastGFile(path_to_pb,'rb') as f:
                       graph_def = tf.GraphDef()
                   graph_def.ParseFromString(f.read())
                   sess.graph.as_default()
                   tf.import_graph_def(graph_def, name='')
                   graph_nodes=[n for n in graph_def.node]
                   names = []
                   for t in graph_nodes:
                      names.append(t.name)
                   print(names)
                        
                   graph = tf.get_default_graph()
                    # Now, let's get hold of the op that we can be processed to get the output.
                    # In the original network y_pred is the tensor that is the prediction of the network
                   y_pred = graph.get_tensor_by_name("y_pred:0")

                   ## Let's feed the images to the input placeholders
                   x= graph.get_tensor_by_name("x:0") 

                    # Creating the feed_dict that is required to be fed to calculate y_pred 
                   feed_dict_testing = {x: x_batch}
                   result=sess.run(y_pred, feed_dict=feed_dict_testing)
                    # Result is of this format [[probabiliy_of_classA probability_of_classB ....]]
                    # Convert np.array to list
                   a = result[0].tolist()
                   text = classes[a.index(max(a))]
                   print("Result",text)
          
except Exception as e:
    print("Exception:",e)

# Calculate execution time
end = time.time()
dur = end-start
print("")
if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")

