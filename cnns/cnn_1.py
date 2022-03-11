# inital simple cnn
#
# Inspired from: https://keras.io/examples/vision/3D_image_classification/
#


from gc import callbacks
import sys
import os
import cv2 as cv
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras

def build_model(width=512, height=512, depth=9):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input(shape=(depth,width,height,1))

    x = keras.layers.Conv3D(filters=16, kernel_size=(1,3,3), activation="relu")(inputs)
    x = keras.layers.MaxPool3D(pool_size=(1,2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    print(x.shape)

    x = keras.layers.Conv3D(filters=16, kernel_size=(2,3,3), activation="relu")(x)
    x = keras.layers.MaxPool3D(pool_size=(1,2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    print(x.shape)

    x = keras.layers.Conv3D(filters=32, kernel_size=(2,3,3), activation="relu")(x)
    x = keras.layers.MaxPool3D(pool_size=(2,2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    print(x.shape)

    x = keras.layers.Conv3D(filters=32, kernel_size=(2,3,3), activation="relu")(x)
    x = keras.layers.MaxPool3D(pool_size=(1,2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    print(x.shape)

    x = keras.layers.Conv3D(filters=64, kernel_size=(2,3,3), activation="relu")(x)
    x = keras.layers.MaxPool3D(pool_size=(1,2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    print(x.shape)

    x = keras.layers.GlobalAveragePooling3D()(x)
    x = keras.layers.Dense(units=625, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    print(x.shape)

    outputs = keras.layers.Dense(units=5, activation="sigmoid")(x)
    print(outputs.shape)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def loss_func(y_true,y_pred):
    loss = keras.backend.mean(keras.backend.square(y_true-y_pred))
    return loss


def load_data(path,xml_root,shuffle=True):
    labels = []
    targets = []
    for item in xml_root:
        if item.tag[:6] == 'sample':
            samplenr = item.tag[7:]
            for subitem in item:
                if subitem.tag == 'specifications':
                    label = [float(subitem.find('pitch').text), float(subitem.find('yaw').text), float(subitem.find('roll').text),
                            float(subitem.find('x_transl').text), float(subitem.find('z_transl').text)]
                    labels.append(label)
                elif subitem.tag == 'images':
                    img3d = []
                    for image in subitem:
                        img_path = os.path.join(path,image.attrib['file'])
                        img = cv.imread(img_path,-1)
                        img3d.append(normalize(img))
                    img3d = np.array(img3d)
                    targets.append(img3d)
    labels = np.array(labels)
    targets = np.array(targets)
    if shuffle:
        perm = np.random.permutation(targets.shape[0])
        targets = targets[perm,:,:,:]
        labels = labels[perm,:]
    return targets, labels

def normalize(img):
    if img.max() > 0:
        norm_img = img.astype(np.float64)
        norm_img = norm_img/norm_img.max()
        return norm_img
    else:
        return img.astype(np.float64)


def main():
    train = True #CHANGE TO FALSE TO EVALUATE MODEL

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--timestp",default='/home/emiwin/exjobb/03071514' ,help=" path to timestamp data folder")
    args = parser.parse_args()

    images = os.path.join(args.timestp,'images')
    xml = os.path.join(args.timestp,'data.xml')
    tree = ET.parse(xml)
    root = tree.getroot()
    targets,labels = load_data(images,root,shuffle=False)


    model = build_model(width=512, height=512, depth=9)
    model.summary() 

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), 
                    loss=loss_func)

    my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=3),
                    tf.keras.callbacks.TensorBoard(log_dir='./logs'),]

    created_dir = False
    n = 1                

    if train:
        model.fit(targets,labels,batch_size=20,epochs=30,shuffle=True,validation_split=0.2,callbacks=my_callbacks)
        #save weights
        while not created_dir:
            try:
                models = os.path.join(args.timestp,'models_{}'.format(n))
                os.mkdir(models)
                created_dir = True
            except OSError:
                created_dir = False
                n += 1
        
        model.save(os.path.join(models,"CNN_1.h5"))
        model.save_weights(os.path.join(models,'cnn_weights.h5'))
    else:
        model.load_weights(os.path.join(models,'cnn_weights.h5'))

        preds = model.predict(targets)

        for n in range(10):
            for i in range(9):
                img = targets[n,i,:,:,:]
                
                if img.max() > 0:
                    img = img*65535.0/img.max() 
                img = img.astype(np.uint16)
                cv.imshow('Input',img)
                print(labels[n,:])
                print(preds[n,:])
                cv.waitKey(0)
                

    

if __name__ == '__main__':
    main()