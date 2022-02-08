import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import os
import shutil
import random
import matplotlib.pyplot as plt
import glob

def train(bat_size, epochh, lrr, val_size, test_size):

    # to make use of gpu accelartion for cuda enabled GPUs
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # creating a valid and test file to use them in calculatig acuracy later 
    if os.path.isdir('dataset/valid') is False:
        os.makedirs('dataset/valid')
        os.makedirs('dataset/valid/without_mask')
        os.makedirs('dataset/valid/with_mask')
        os.makedirs('dataset/test')
        os.makedirs('dataset/test/without_mask')
        os.makedirs('dataset/test/with_mask')

        # moving 400 of our train sample to valdation sample
        for c in random.sample(glob.glob('dataset/train/with_mask/*'), val_size):
            shutil.move(c, 'dataset/valid/with_mask' )
        for c in random.sample(glob.glob('dataset/train/without_mask/*'), val_size):
            shutil.move(c, 'dataset/valid/without_mask' )
        #moving 200 of our traning sample to our test sample
        for c in random.sample(glob.glob('dataset/train/with_mask/*'), test_size):
            shutil.move(c, 'dataset/test/with_mask' )
        for c in random.sample(glob.glob('dataset/train/without_mask/*'), test_size):
            shutil.move(c, 'dataset/test/without_mask' )    

    train_dir = "dataset/train"
    val_dir = "dataset/val"


    train_batches = ImageDataGenerator(rotation_range=5,zoom_range=0.15,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.10,horizontal_flip=True,fill_mode="nearest",preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input) \
        .flow_from_directory(directory=train_dir, target_size=(224,224), classes=['with_mask', 'without_mask'], batch_size=bat_size)

    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input) \
        .flow_from_directory(directory=train_dir, target_size=(224,224), classes=['with_mask', 'without_mask'], batch_size=bat_size)
        
    mobile = tf.keras.applications.mobilenet.MobileNet()
    x = mobile.layers[-6].output
    output = Dense(units=2, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=output)
    
    for layer in model.layers[:-25]:
        layer.trainable = False

        

    model.compile(optimizer=Adam(lr=lrr), loss='binary_crossentropy', metrics=['accuracy'])


    fit_function = model.fit(x=train_batches,
            steps_per_epoch=len(train_batches),
            validation_data=valid_batches,
            validation_steps=len(valid_batches),
            epochs=epochh,
        
    )

    model.save('mask_model.h5')
    print(fit_function.history)

    #plotting and saving traning graph to use it in graph.py
    N = 5
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), fit_function.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), fit_function.history["accuracy"], label="train_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('images/train.png')


    #plotting val acuracy and loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), fit_function.history["val_loss"], label="Validation_loss")
    plt.plot(np.arange(0, N), fit_function.history["val_accuracy"], label="Validation_acc")
    plt.title("Validation Loss and Accuracy")
    plt.xlabel("Epoch ")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('images/val.png')



