import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import cv2
import sys
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf
import os
#this confegration is used to make use of cuda gpu and accelrate the prosses of of detection.

def prepare_image(face_image):

    face_image1 = cv2.resize(face_image, (224, 224)) # resizeing the image to 224 by 224 because the moblilenext excpect this size
    rgb_face = cv2.cvtColor(face_image1, cv2.COLOR_BGR2RGB) # 
    img_array = keras.preprocessing.image.img_to_array(rgb_face)
    return tf.keras.applications.mobilenet.preprocess_input(img_array)




def webcam(vid_path):

# to make use of gpu accelartion for cuda enabled GPUs
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    is_vid = os.path.isfile(vid_path)
    if not is_vid:
        print("video path is not correct!")
        

    


    deplay = 'caffe_face/deploy.prototxt'
    weight = "caffe_face/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNet(deplay, weight)
    is_deplay = os.path.isfile(deplay)
    is_wight = os.path.isfile (weight)
    if not is_wight or not is_deplay:
        print('cannot locate the model')


    # load the face mask detector model from disk
    model = keras.models.load_model("mask_model.h5")
    #opening the webcam
    video_capture = cv2.VideoCapture(vid_path)
    ret = True


    while ret:

        
        if not video_capture.isOpened():
            print('Unable to load the video/camera.')
            exit()

        # Capture frame-by-frame
        ret, image = video_capture.read()
        if not ret:
            break
        (h, w) = image.shape[:2]

        #  resize_process the image to fit in the caffe model
        resize_process = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                            (104.0, 177.0, 123.0))

        # pass the resized imagae into the net to get the faces.
        net.setInput(resize_process)
        faces = net.forward()
        

        # loop over the faces
        
        for i in range(0, faces.shape[2]):
            # calculte the probalbilty for evey face obtained from the net
            probability = faces[0, 0, i, 2]

            # remove low probablilty faces and construct a box
            if probability > 0.6:

                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (X, Y, X_w, Y_h) = box.astype("int")

                (X, Y) = (max(0, X), max(0, Y))
                (X_w, Y_h) = (min(w - 1, X_w), min(h - 1, Y_h))

                # preprosses the image to get it ready to pass it to the fined tuned mobilenet network
                
                face = image[Y:Y_h, X:X_w]
                if face.shape[0]==0 or face.shape[1]==0:
                    continue
                face_array = prepare_image(face)
                face_array = np.expand_dims(face_array, axis=0)
                predict_array = model.predict(face_array)
                (mask, withoutMask) = predict_array[0]

                # determine the class text and color we'll use to draw
                # the bounding box and text
                text = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 254, 254) if text == "No Mask" else (255, 0, 0)

                # include the probability in the text
                text = "{}: {:.2f}%".format(text, max(mask, withoutMask) * 100)

            
                cv2.putText(image, text, (X, Y - 10),
                            cv2.QT_FONT_NORMAL, 0.35, color, 2)
                cv2.rectangle(image, (X, Y), (X_w, Y_h), color, 2)


            # show the output image
        cv2.imshow("Output", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
    video_capture.release()
    cv2.destroyAllWindows()




