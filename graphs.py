import numpy as np
import cv2
import os
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot as plt
import itertools
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def test_and_graph(dic_pat):
    # to make use of gpu accelartion for cuda enabled GPUs
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


    mymodel = keras.models.load_model('mask_model.h5')
    test_batches = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).\
    flow_from_directory(directory=dic_pat, target_size=(224,224), batch_size=10, shuffle=False)
    predictions = mymodel.predict(x=test_batches, steps=len(test_batches), verbose=0)
    test_labels = test_batches.classes
    cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
    cm_plot_labels = ['mask','no mask']
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')





def train_acuracy_loss():
    img = mpimg.imread('images/train.png')
    imgplot = plt.imshow(img)
    plt.show()
    

def val_acc_loss():
    img = mpimg.imread('images/val.png')
    imgplot = plt.imshow(img)
    plt.show()
    
