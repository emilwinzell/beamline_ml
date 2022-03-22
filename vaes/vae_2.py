#
# Inspired by: https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/
#
#
#  Version 2.0
#  Trained for 100 epochs 20 bsize, still very poor results...
#
#
# Version 2.1 added generator for hadling big datsets
# trying to train on simple ellipses
# Same results as previously, all values in decoded negative??? Turns out it was heavily overtrained...
#
# Version 2.2 latent_space = 10
# Version 2.3 Early stopping, new rec loss factor
# Version 2.4 New loss function
import sys
sys.stdout = open('output.txt','wt')
import os
import cv2 as cv
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence

import glob

import logging
logging.basicConfig(filename='./vae_2.log', level=logging.DEBUG)
logger=logging.getLogger(__name__)

def load_ellipses(path):
    list_of_imgs = glob.glob(os.path.join(path,'*.png'))
    targets = []
    img3d = []
    for i,img_name in enumerate(list_of_imgs[:18]):
        img = cv.imread(img_name,0)
        img3d.append(img)
        if len(img3d) == 9:
            targets.append(img3d)
            img3d = []
    targets = np.array(targets)
    return targets

class My_Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.x, self.y = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size) / 9))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size*9:(idx + 1) * self.batch_size*9]
        batch_y = self.y[idx * self.batch_size*9:(idx + 1) * self.batch_size*9]

        targets = []
        img3d = []
        for img_name in batch_x:
            img = normalize(cv.imread(img_name,-1))
            img3d.append(img)
            if len(img3d) == 9:
                targets.append(img3d)
                img3d = []
        return np.array(targets), np.array(targets)

def build_encoder(width=512,height=512,depth=9,latent_space_dim=5):
    print('ENCODER')
    encoder_input = keras.layers.Input(shape=(depth,width,height,1), name='encoder_input')

    x = keras.layers.Conv3D(filters=1, kernel_size=3, strides=(1,1,1), name='encoder_conv_1')(encoder_input)
    x = keras.layers.BatchNormalization(name='encoder_norm_1')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_1')(x)
    print(x.shape)

    x = keras.layers.Conv3D(filters=8, kernel_size=3, strides=(1,1,1), name='encoder_conv_2')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_2')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_2')(x)
    print(x.shape)

    x = keras.layers.Conv3D(filters=8, kernel_size=3, strides=(1,1,1), name='encoder_conv_3')(x)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_3')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_3')(x)
    print(x.shape)

    x = keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, name='encoder_conv_4')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_4')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_4')(x)
    print(x.shape)

    x = keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=2, name='encoder_conv_5')(x)
    x = keras.layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='valid')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_5')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_5')(x)
    print(x.shape)

    x = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, name='encoder_conv_6')(x)
    x = keras.layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='valid')(x)
    x = keras.layers.BatchNormalization(name='encoder_norm_6')(x)
    x = keras.layers.LeakyReLU(name='encoder_leayrelu_6')(x)
    print(x.shape)

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
    print('DECODER')
    decoder_input = keras.layers.Input(shape=(latent_space_dim), name="decoder_input")

    x = keras.layers.Dense(units=np.prod(shape), name="decoder_dense_1")(decoder_input)
    x = keras.layers.Reshape(target_shape=shape)(x)
    print(x.shape)

    x = keras.layers.Conv3DTranspose(filters=32, kernel_size=(2,4,4),padding='valid',strides=(2,4,4), name="decoder_conv_tran_1")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_1')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_1')(x)
    print(x.shape)

    x = keras.layers.Conv3DTranspose(filters=16, kernel_size=(2,3,3),padding='valid', strides=(2,2,2), name="decoder_conv_tran_2")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_2')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_2')(x)
    print(x.shape)

    x = keras.layers.Conv3DTranspose(filters=16, kernel_size=(1,3,3),padding='valid',strides=(2,2,2), name="decoder_conv_tran_3")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_3')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_3')(x)
    print(x.shape)

    x = keras.layers.Conv3DTranspose(filters=8, kernel_size=(1,2,2),padding='valid',strides=(1,2,2), name="decoder_conv_tran_4")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_4')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_4')(x)
    print(x.shape)

    x = keras.layers.Conv3DTranspose(filters=8, kernel_size=(1,13,13),padding='valid',strides=(1,1,1), name="decoder_conv_tran_5")(x)
    x = keras.layers.BatchNormalization(name='decoder_norm_5')(x)
    x = keras.layers.LeakyReLU(name='decoder_leayrelu_5')(x)
    print(x.shape)

    x = keras.layers.Conv3DTranspose(filters=1, kernel_size=(2,15,15),padding='valid',strides=(1,1,1), name="decoder_conv_tran_6")(x)
    decoder_output = keras.layers.LeakyReLU(name="decoder_output")(x)
    print(x.shape)

    model = keras.models.Model(decoder_input, decoder_output, name="decoder_model")
    return model

def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 10e6
        reconstruction_loss = keras.backend.mean(keras.backend.square(y_true-y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * keras.backend.sum(1.0 + encoder_log_variance - keras.backend.square(encoder_mu) - keras.backend.exp(encoder_log_variance), axis=[1,2,3])
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
        norm_img = img.astype(np.float32)
        norm_img = norm_img/norm_img.max()
        return norm_img
    else:
        return img.astype(np.float32)

def main():
    train = True #CHANGE TO FALSE TO EVALUATE MODEL

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--timestp",default='/home/emiwin/exjobb/ellipses' ,help=" path to timestamp data folder")
    args = parser.parse_args()

    images = os.path.join(args.timestp,'images')
    #xml = os.path.join(args.timestp,'data.xml')
    #tree = ET.parse(xml)
    #root = tree.getroot()
    #targets,labels = load_data(images,root)

    list_of_imgs = glob.glob(os.path.join(images,'*.png'))
    

    if len(list_of_imgs)%9 != 0 or len(list_of_imgs) == 0:
        print('DATA ERROR, wrong length, ending...')
        return

    num_samples = len(list_of_imgs)//9
    num_train = int(num_samples*0.8) # 80% to train
    x_train = list_of_imgs[:num_train*9]
    x_test = list_of_imgs[num_train*9:]
    labels = np.ones((num_samples,5),dtype=np.float32)
    y_train = labels[:num_train,:]
    y_test = labels[num_train:,:]

    print('loaded {} samples'.format(num_samples))

    
    latent_space_dim = 10
    img_size = (512,512)
    depth = 9
    encoder,enc_mu,enc_log_var, shape = build_encoder(width=img_size[0],height=img_size[1],depth=depth,latent_space_dim=latent_space_dim)
    encoder.summary()

    decoder = build_decoder(shape,latent_space_dim=latent_space_dim)
    decoder.summary()

    vae_input = keras.layers.Input(shape=(depth,img_size[0], img_size[1], 1), name="VAE_input")
    vae_encoder_output = encoder(vae_input)

    vae_decoder_output = decoder(vae_encoder_output)
    vae = keras.models.Model(vae_input, vae_decoder_output, name="VAE")
    vae.summary()

    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=loss_func(enc_mu, enc_log_var))
    models = os.path.join(args.timestp,'models_3')   
    created_dir = False
    n = 1

    # Batch generators:
    batch_size = 10
    my_training_batch_generator = My_Generator(x_train, y_train, batch_size)
    my_validation_batch_generator = My_Generator(x_test, y_test, batch_size)


    if train:
        while not created_dir:
            try:
                os.mkdir(models)
                created_dir = True
            except OSError:
                created_dir = False
                n += 1
                models = os.path.join(args.timestp,'models_{}'.format(n))
        
        print('STARTING TRAINING')
        # Callbacks:
        tb = keras.callbacks.TensorBoard(log_dir=os.path.join(models,'logs'), histogram_freq=0, write_graph=True, write_images=True)
        es = keras.callbacks.EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True)
        try:
            #vae.fit(x_train, x_train, epochs=100, batch_size=20, shuffle=True, validation_data=(x_test, x_test))
            vae.fit(my_training_batch_generator,
                    steps_per_epoch=(num_train // batch_size),
                    epochs=15,
                    verbose=1,
                    validation_data=my_validation_batch_generator,
                    validation_steps=((num_samples-num_train) // batch_size),
                    callbacks=[tb,es])
        except Exception as e:
            logger.error(e)

        
        encoder.save(os.path.join(models,"VAE_encoder.h5")) 
        decoder.save(os.path.join(models,"VAE_decoder.h5"))
        vae.save(os.path.join(models,"VAE.h5"))
        encoder.save_weights(os.path.join(models,'encoder_weights.h5'))
        decoder.save_weights(os.path.join(models,'decoder_weights.h5'))
    else:
        encoder.load_weights(os.path.join(models,'encoder_weights.h5'))
        decoder.load_weights(os.path.join(models,'decoder_weights.h5'))
        
        (x_test,_) = my_validation_batch_generator.__getitem__(0)
        print(x_test.shape)
        
        encoded_data = encoder.predict(x_test)
        decoded_data = decoder.predict(encoded_data)

        #print(y_test)
        print(decoded_data.shape)

        for n in range(10):
            for i in range(9):
                img = decoded_data[n,i,:,:,:]
                org = x_test[n,i,:,:]
                #img[img < 0] = 0
                img = abs(img)
                print(img.max())
                img = img*255/img.max()
                org = org*255
                img = img.astype(np.uint8)
                org = org.astype(np.uint8)
                print(org.max())
                cv.imshow('Decoded',img)
                cv.imshow('Org',org)
                cv.waitKey(0)


if __name__ == '__main__':
    main()
