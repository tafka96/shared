import tensorflow as tf
import sys
import os
import numpy as np
import facenet
import align.detect_face
from scipy import misc

image_path="Andres_Herkel_0001.jpg"
#image_path="meeting.jpg"
margin=44
image_size=182

with tf.Graph().as_default():        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
random_key = np.random.randint(0, high=99999)
bounding_boxes_filename = os.path.join("", 'bounding_boxes_%05d.txt' % random_key)


img = misc.imread(image_path)
bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
 
nrof_faces = bounding_boxes.shape[0]

if nrof_faces>0:
    det_arr = []
    img_size = np.asarray(img.shape)[0:2]
    for i in range(nrof_faces):
        det_arr.append(np.squeeze(bounding_boxes[i]))
        

    number = 1
    for i, det in enumerate(det_arr):
            
        det = np.squeeze(det)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        output="output"+str(number)+".png"
        number=number+1
        misc.imsave(output, scaled)
        #text_file.write('%s %d %d %d %d\n' % (output, bb[0], bb[1], bb[2], bb[3]))
            
            
