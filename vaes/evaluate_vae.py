# evaluate 2d vae
#
# Trying differnet implementation
# from: https://keras.io/examples/generative/vae/?msclkid=0a8e33e0a5ff11ecab29a7e68ba38092
#
# Adding second model for scaling distances.
# file:///C:/Users/emiwin/Downloads/Variational_Autoencoders_for_Cancer_Data_Integrati.pdf
#
import sys
sys.stdout = open('output.txt','wt')
import os
import cv2 as cv
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from tensorflow import keras
import glob
import random
from scipy import optimize

import logging
logging.basicConfig(filename='./simple_vae_4.log', level=logging.DEBUG)
logger=logging.getLogger(__name__)

from simple_vae_4 import My_Generator, VAE



def build_encoder(width=256,height=256,latent_space_dim=10):
    print('ENCODER')
    encoder_input = keras.layers.Input(shape=(width,height,1), name='encoder_input')
    ex_input = keras.layers.Input(shape=(4,), name='extra_input')
    y = keras.layers.Dense(4,activation='relu')(ex_input)

    x = keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=1, name='encoder_conv_1')(encoder_input)
    x = keras.layers.BatchNormalization(name='encoder_norm_1')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_1')(x)
    print(x.shape)

    x = keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=1, name='encoder_conv_2')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_2')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_2')(x)
    print(x.shape)

    x = keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=1, name='encoder_conv_3')(x)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_3')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_3')(x)
    print(x.shape)

    x = keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, name='encoder_conv_4')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_4')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_4')(x)
    print(x.shape)

    x = keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=2, name='encoder_conv_5')(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_5')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_5')(x)
    print(x.shape)

    x = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, name='encoder_conv_6')(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_6')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_6')(x)
    print(x.shape)

    shape_before_flatten = keras.backend.int_shape(x)[1:]
    enc_flatten = keras.layers.Flatten()(x)
    x = keras.layers.Dense(np.prod(shape_before_flatten),activation="relu")(enc_flatten)
    shared_input = keras.layers.Concatenate()([x,y])

    encoder_mu = keras.layers.Dense(units=latent_space_dim, name="encoder_mu")(shared_input)
    encoder_log_variance = keras.layers.Dense(units=latent_space_dim, name="encoder_log_variance")(shared_input)

    z = Sampling()([encoder_mu,encoder_log_variance])

    model = keras.models.Model([encoder_input,ex_input], [encoder_mu, encoder_log_variance, z], name="encoder_model")

    return model,shape_before_flatten


def build_decoder(shape,latent_space_dim=5):
    print('DECODER')
    decoder_input = keras.layers.Input(shape=(latent_space_dim,), name="decoder_input")

    x = keras.layers.Dense(units=np.prod(shape), name="decoder_dense_1")(decoder_input)
    x = keras.layers.Reshape(target_shape=shape)(x)
    print(x.shape)

    x = keras.layers.Conv2DTranspose(filters=32, kernel_size=(3,3),padding='valid',strides=2, name="decoder_conv_tran_1")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_1')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_1')(x)
    print(x.shape)

    x = keras.layers.Conv2DTranspose(filters=16, kernel_size=(3,3),padding='valid', strides=2, name="decoder_conv_tran_2")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_2')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_2')(x)
    print(x.shape)

    x = keras.layers.Conv2DTranspose(filters=16, kernel_size=(3,3),padding='valid',strides=2, name="decoder_conv_tran_3")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_3')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_3')(x)
    print(x.shape)

    x = keras.layers.Conv2DTranspose(filters=8, kernel_size=(3,3),padding='same',strides=2, name="decoder_conv_tran_4")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_4')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_4')(x)
    print(x.shape)

    x = keras.layers.Conv2DTranspose(filters=8, kernel_size=(3,3),padding='valid',strides=2, name="decoder_conv_tran_5")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_5')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_5')(x)
    print(x.shape)

    x = keras.layers.Conv2DTranspose(filters=1, kernel_size=(4,4),padding='valid',strides=1, name="decoder_conv_tran_6")(x)
    x = keras.layers.LeakyReLU(name="decoder_output")(x)
    print(x.shape)

    y = keras.layers.Dense(4,name="decoder_dense_ex_1")(decoder_input)
    y = keras.layers.Dense(4,name="decoder_dense_ex_2")(y)
    print(y.shape)

    model = keras.models.Model(decoder_input, [x,y], name="decoder_model")
    return model


def calculate_fwhm(data,bins):
        n = np.count_nonzero(data==0)//2+1
        est_std = (bins[-n]-bins[n])/5.0
        est_mean = bins[len(bins)//2]
        est_amp = np.max(data)*2.5
        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-((x - mean) / np.sqrt(2) / stddev)**2)
        
        [amp,mean,std],_ = optimize.curve_fit(gaussian, bins, data,p0=[est_amp,est_mean,est_std])
        std = abs(std)
        fwhm = 2*np.sqrt(2*np.log(2))*std
        return fwhm


def evaluate(img,scale):
    x1D = np.sum(img, axis=0)
    y1D = np.sum(img, axis=1)
    cx,dx,cy,dy = scale

    
    xbins = np.linspace(cx-dx/2,cx+dx/2,len(x1D))
    ybins = np.linspace(cy-dy/2,cy+dy/2,len(y1D))

    fwhmX = calculate_fwhm(x1D,xbins)
    fwhmY = calculate_fwhm(y1D,ybins)
    return fwhmX, fwhmY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",default='/home/emiwin/exjobb/' ,help="path to model")
    parser.add_argument("-t", "--timestp", default='/home/emiwin/exjobb/', help="path to timestamp folder")
    args = parser.parse_args()

    latent_space_dim = 10
    img_size = (256,256)
    encoder, shape = build_encoder(width=img_size[0],height=img_size[1],latent_space_dim=latent_space_dim)
    encoder.summary()

    decoder = build_decoder(shape,latent_space_dim=latent_space_dim)
    decoder.summary()

    histograms = os.path.join(args.base,'histograms')
    images = os.path.join(args.timestp,'images')
    xml = os.path.join(args.timestp,'data.xml')
    tree = ET.parse(xml)
    root = tree.getroot()
    
    test_imgs = glob.glob(os.path.join(images,'*.png'))
    num_samples = len(test_imgs)
    
    random.shuffle(test_imgs)
    print('loaded {} samples'.format(num_samples))
    
    
    encoder.load_weights(os.path.join(args.model,'encoder_weights.h5'))
    decoder.load_weights(os.path.join(args.model,'decoder_weights.h5'))

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),run_eagerly=False)

    # Batch generators:
    batch_size = num_samples//5
    my_test_batch_generator = My_Generator(test_imgs, root, batch_size)
    
    for i in range(num_samples//batch_size):
        x_test = my_test_batch_generator.__getitem__(i)
        
        input_imgs = x_test[0]
        input_scales = x_test[1]

        encoded_data = encoder.predict(x_test)
        [output_imgs,output_scales] = decoder.predict(encoded_data[2])
        
        xdiff=[]
        ydiff=[]
        for i_img,i_sc,o_img,o_sc in zip(input_imgs,input_scales,output_imgs,output_scales):
            tx,ty = evaluate(i_img,i_sc)
            px,py = evaluate(o_img,o_sc)

            xdiff.append(abs(tx-px))
            ydiff.append(abs(ty-py))

        print('Average batch FWHM errors: lateral={}, vertical={}'.format(np.mean(xdiff),np.mean(ydiff)))




if __name__ == '__main__':
    main()
