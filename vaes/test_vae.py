#
# Inspired by: https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/
#

import sys
import os
import cv2 as cv
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras




def load_data(path,xml_root):
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
    return targets, labels

def normalize(img):
    if img.max() > 0:
        norm_img = img.astype(np.float64)
        norm_img = norm_img/norm_img.max()
        return norm_img
    else:
        return img.astype(np.float64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--timestp",default='/home/emiwin/exjobb/??' ,help=" path to timestamp data folder")
    args = parser.parse_args()

    images = os.path.join(args.timestp,'images')
    xml = os.path.join(args.timestp,'data.xml')
    tree = ET.parse(xml)
    root = tree.getroot()
    targets,labels = load_data(images,root)
    x_train = targets[:8,:,:,:]
    x_test = targets[8:,:,:,:]
    y_test = labels[8:,:]

    

    models = os.path.join(args.timestp,'models')
    

    encoder = keras.models.load_model(os.path.join(models,"VAE_encoder.h5")) 
    decoder = keras.models.load_model(os.path.join(models,"VAE_decoder.h5"))

    encoder.save_weights(os.path.join(models,'encoder_weights.h5'))
    decoder.save_weights(os.path.join(models,'decoder_weights.h5'))

    #encoded_data = encoder.predict(x_test)
    #decoded_data = decoder.predict(encoded_data)

    #print(y_test)
    #print(encoded_data)

    

if __name__ == '__main__':
    main()