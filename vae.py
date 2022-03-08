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


def build_encoder(width=512,height=512,depth=9,latent_space_dim=5):
    #print('ENCODER')
    encoder_input = keras.layers.Input(shape=(depth,width,height,1), name='encoder_input')

    x = keras.layers.Conv3D(filters=1, kernel_size=3, strides=(1,1,1), name='encoder_conv_1')(encoder_input)
    x = keras.layers.BatchNormalization(name='encoder_norm_1')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_1')(x)
    #print(x.shape)

    x = keras.layers.Conv3D(filters=4, kernel_size=3, strides=(1,1,1), name='encoder_conv_2')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_2')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_2')(x)
    #print(x.shape)

    x = keras.layers.Conv3D(filters=8, kernel_size=3, strides=(2,2,2), name='encoder_conv_3')(x)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_3')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_3')(x)
    #print(x.shape)

    x = keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=2, name='encoder_conv_4')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_4')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_4')(x)
    #print(x.shape)

    x = keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=2, name='encoder_conv_5')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_5')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_5')(x)
    #print(x.shape)

    shape_before_flatten = keras.backend.int_shape(x)[1:]
    enc_flatten = keras.layers.Flatten()(x)

    encoder_mu = keras.layers.Dense(units=latent_space_dim, name="encoder_mu")(enc_flatten)
    encoder_log_variance = keras.layers.Dense(units=latent_space_dim, name="encoder_log_variance")(enc_flatten)

    encoder_mu_log_variance_model = keras.models.Model(x, (encoder_mu, encoder_log_variance), name="encoder_mu_log_variance_model")

    def sampling(mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = keras.backend.random_normal(shape=keras.backend.shape(mu), mean=0.0, stddev=1.0)
        random_sample = mu + keras.backend.exp(log_variance/2) * epsilon
        return random_sample

    encoder_output = keras.layers.Lambda(sampling, name="encoder_output")([encoder_mu, encoder_log_variance])

    model = keras.models.Model(encoder_input, encoder_output, name="encoder_model")
    return model,encoder_mu,encoder_log_variance,shape_before_flatten

def build_decoder(shape,latent_space_dim=5):
    #print('DECODER')
    decoder_input = keras.layers.Input(shape=(latent_space_dim), name="decoder_input")

    x = keras.layers.Dense(units=np.prod(shape), name="decoder_dense_1")(decoder_input)
    x = keras.layers.Reshape(target_shape=shape)(x)
    #print(x.shape)

    x = keras.layers.Conv3DTranspose(filters=8, kernel_size=(2,4,4),padding='valid',strides=(2,2,2), name="decoder_conv_tran_1")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_1')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_1')(x)
    #print(x.shape)

    x = keras.layers.Conv3DTranspose(filters=8, kernel_size=(2,3,3),padding='valid', strides=(2,2,2), name="decoder_conv_tran_2")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_2')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_2')(x)
    #print(x.shape)

    x = keras.layers.Conv3DTranspose(filters=4, kernel_size=1,padding='valid',strides=(2,2,2), name="decoder_conv_tran_3")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_3')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_3')(x)
    #print(x.shape)

    x = keras.layers.Conv3DTranspose(filters=4, kernel_size=(1,4,4),padding='valid',strides=(1,2,2), name="decoder_conv_tran_4")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_4')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_4')(x)
    #print(x.shape)

    x = keras.layers.Conv3DTranspose(filters=1, kernel_size=(2,11,11),padding='valid',strides=(1,1,1), name="decoder_conv_tran_5")(x)
    decoder_output = keras.layers.LeakyReLU(name="decoder_output")(x)
    #print(x.shape)

    model = keras.models.Model(decoder_input, decoder_output, name="decoder_model")
    return model

def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = keras.backend.mean(keras.backend.square(y_true-y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        print(encoder_mu.shape)
        print(encoder_log_variance.shape)
        kl_loss = -0.5 * keras.backend.sum(1.0 + encoder_log_variance - keras.backend.square(encoder_mu) - keras.backend.exp(encoder_log_variance), axis=[1,2,3])
        print(kl_loss.shape)
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * keras.backend.sum(1.0 + encoder_log_variance - keras.backend.square(encoder_mu) - keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)
        # print(y_true.shape)
        # print(y_predict.shape)
        # print(reconstruction_loss.shape)
        # print(kl_loss.shape)
        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss

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

    
    latent_space_dim = 5
    img_size = (512,512)
    depth = 9
    encoder_model,enc_mu,enc_log_var, shape = build_encoder(width=img_size[0],height=img_size[1],depth=depth,latent_space_dim=latent_space_dim)
    encoder_model.summary()

    decoder_model = build_decoder(shape,latent_space_dim=latent_space_dim)
    decoder_model.summary()

    vae_input = keras.layers.Input(shape=(depth,img_size[0], img_size[1], 1), name="VAE_input")
    vae_encoder_output = encoder_model(vae_input)

    vae_decoder_output = decoder_model(vae_encoder_output)
    vae = keras.models.Model(vae_input, vae_decoder_output, name="VAE")
    vae.summary()

    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=loss_func(enc_mu, enc_log_var))


    vae.fit(x_train, x_train, epochs=4, batch_size=8, shuffle=True, validation_data=(x_test, x_test))
    

if __name__ == '__main__':
    main()