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
    # depth is 9*3
    encoder_input = keras.layers.Input(shape=(depth,width,height,3), name='encoder_input')

    x = keras.layers.Conv3D(filters=1, kernel_size=(3,3,3), strides=(1,1,1), name='encoder_conv_1')(encoder_input)
    x = keras.layers.BatchNormalization(name='encoder_norm_1')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_1')(x)

    x = keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), strides=(1,1,1), name='encoder_conv_2')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_2')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_2')(x)

    x = keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), strides=(2,2,2), name='encoder_conv_3')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_3')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_3')(x)

    x = keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), strides=(2,2,2), name='encoder_conv_4')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_4')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_4')(x)

    x = keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), strides=(1,1,1), name='encoder_conv_5')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_5')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_5')(x)

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
    decoder_input = keras.layers.Input(shape=(latent_space_dim), name="decoder_input")

    x = keras.layers.Dense(units=np.prod(shape), name="decoder_dense_1")(decoder_input)
    x = keras.layers.Reshape(target_shape=shape)(x)

    x = keras.layers.Conv3DTranspose(filters=64, kernel_size=3, strides=1, name="decoder_conv_tran_1")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_1')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_1')(x)

    x = keras.layers.Conv3DTranspose(filters=64, kernel_size=3, strides=2, name="decoder_conv_tran_2")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_2')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_2')(x)

    x = keras.layers.Conv3DTranspose(filters=64, kernel_size=3, strides=2, name="decoder_conv_tran_3")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_3')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_3')(x)

    x = keras.layers.Conv3DTranspose(filters=1, kernel_size=3,  strides=1, name="decoder_conv_tran_4")(x)
    decoder_output = keras.layers.LeakyReLU(name="decoder_output")(x)

    model = keras.models.Model(decoder_input, decoder_output, name="decoder_model")
    return model

def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = keras.backend.mean(keras.backend.square(y_true-y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * keras.backend.sum(1.0 + encoder_log_variance - keras.backend.square(encoder_mu) - keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * keras.backend.sum(1.0 + encoder_log_variance - keras.backend.square(encoder_mu) - keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss


def main():
    latent_space_dim = 5
    img_size = (512,512,3)
    depth = 9
    encoder_model,enc_mu,enc_log_var, shape = build_encoder(width=img_size[0],height=img_size[1],depth=depth,latent_space_dim=latent_space_dim)
    encoder_model.summary()

    decoder_model = build_decoder(shape,latent_space_dim=latent_space_dim)
    decoder_model.summary()

    vae_input = keras.layers.Input(shape=(img_size[0], img_size[1], depth), name="VAE_input")
    vae_encoder_output = encoder_model(vae_input)

    vae_decoder_output = decoder_model(vae_encoder_output)
    vae = keras.models.Model(vae_input, vae_decoder_output, name="VAE")
    vae.summary()

    vae.compile(optimizer=keras.optimizers.Adam(lr=0.0005), loss=loss_func(enc_mu, enc_log_var))



if __name__ == '__main__':
    main()