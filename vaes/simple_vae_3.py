# vae_3
#
# Trying differnet implementation
# from: https://keras.io/examples/generative/vae/?msclkid=0a8e33e0a5ff11ecab29a7e68ba38092
#
# Still nothing in the decoded images
#
import sys
#sys.stdout = open('output.txt','wt')
import os
import cv2 as cv
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
import glob
import random
import math

import logging
logging.basicConfig(filename='./simple_vae_3.log', level=logging.DEBUG)
logger=logging.getLogger(__name__)


class My_Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.x, self.y = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __normalize__(self,img,bytes=8):
        m = float(2**bytes-1)
        if img.max() > 0:
            norm_img = img.astype(np.float32)
            norm_img = norm_img/m
            return norm_img
        else:
            return img.astype(np.float32)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        targets = []
        for img in batch_x:
            #img = self.__normalize__(cv.imread(img_name,0))
            #img = cv.blur(img,(5,5))
            img = cv.resize(img,(256,256)) 
            nimg = self.__normalize__(img)
            targets.append(nimg)
        return np.array(targets)


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def weighted_binary_crossentropy(self,target, output):
        loss = -(85.0 * target * K.log(output) + 15.0 * (1.0 - target) * K.log(1.0 - output)) / 100.0
        return loss

    def __loss_fcn(self,data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction = tf.squeeze(reconstruction)
        reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    self.weighted_binary_crossentropy(data, K.clip(reconstruction, 1e-7, 1.0 - 1e-7)), axis=(1, 2)
                )
            )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
        #reconstruction_loss = K.mean(self.weighted_binary_crossentropy(data, K.clip(reconstruction, 1e-7, 1.0 - 1e-7)))     
        #kl_factor = 1
        #kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))*kl_factor
        total_loss = reconstruction_loss + kl_loss

        return reconstruction_loss, kl_loss, total_loss



    def train_step(self, data):
        print(data.shape)
        with tf.GradientTape() as tape:
            reconstruction_loss, kl_loss, total_loss = self.__loss_fcn(data)
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
      if isinstance(data, tuple):
        data = data[0]

      reconstruction_loss, kl_loss, total_loss = self.__loss_fcn(data)
      return {
          "loss": total_loss,
          "reconstruction_loss": reconstruction_loss,
          "kl_loss": kl_loss,
      }

    #def call(self, inputs, training=None, mask=None):
    #    _,_,z = self.encoder(inputs=inputs, training=training, mask=mask)
    #    return self.decoder(z)


def build_encoder(width=512,height=512,latent_space_dim=5):
    print('ENCODER')
    encoder_input = keras.layers.Input(shape=(width,height,1), name='encoder_input')

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

    encoder_mu = keras.layers.Dense(units=latent_space_dim, name="encoder_mu")(x)
    encoder_log_variance = keras.layers.Dense(units=latent_space_dim, name="encoder_log_variance")(x)

    z = Sampling()([encoder_mu,encoder_log_variance])

    model = keras.models.Model(encoder_input, [encoder_mu, encoder_log_variance, z], name="encoder_model")

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
    decoder_output = keras.layers.LeakyReLU(name="decoder_output")(x)
    print(x.shape)

    model = keras.models.Model(decoder_input, decoder_output, name="decoder_model")
    return model


def normalize(img):
    if img.max() > 0:
        norm_img = img.astype(np.float32)
        norm_img = norm_img/norm_img.max()
        return norm_img
    else:
        return img.astype(np.float32)

def scheduler(epoch):
   epoch = epoch-43
   initial_lrate = 0.01
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

def main():
    train = True #CHANGE TO FALSE TO EVALUATE MODEL

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--timestp",default='/home/emiwin/exjobb/ellipses2' ,help=" path to timestamp data folder")
    args = parser.parse_args()

    images = os.path.join(args.timestp,'images')
    #xml = os.path.join(args.timestp,'data.xml')
    #tree = ET.parse(xml)
    #root = tree.getroot()
    #targets,labels = load_data(images,root)
    """
    list_of_imgs = glob.glob(os.path.join(images,'*.png'))
    

    num_samples = len(list_of_imgs)
    num_train = int(num_samples*0.8) # 80% to train
    x_train = list_of_imgs[:num_train]
    x_test = list_of_imgs[num_train:]
    labels = np.ones((num_samples,5),dtype=np.float32)
    y_train = labels[:num_train,:]
    y_test = labels[num_train:,:]
	
    random.shuffle(x_train)
    len(x_train)
    print('loaded {} samples'.format(num_samples))
    """
    (x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
    num_train = len(x_train)
    num_samples = len(x_test)+num_train
    
    latent_space_dim = 10
    img_size = (256,256)
    encoder, shape = build_encoder(width=img_size[0],height=img_size[1],latent_space_dim=latent_space_dim)
    #encoder.summary()

    decoder = build_decoder(shape,latent_space_dim=latent_space_dim)
    #decoder.summary()


    models = os.path.join(args.timestp,'models_1')   
    created_dir = True
    n = 1

    #encoder.load_weights(os.path.join(models,'encoder_weights4.h5'))
    #decoder.load_weights(os.path.join(models,'decoder_weights4.h5'))

    vae = VAE(encoder, decoder)
    sgd = keras.optimizers.SGD(lr = 0.005, momentum = 0.6, nesterov = True)
    #vae.compile(optimizer=sgd)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),run_eagerly=False)

    # Batch generators:
    batch_size = 200
    my_training_batch_generator = My_Generator(x_train, y_train, batch_size)
    my_validation_batch_generator = My_Generator(x_test, y_test, batch_size)


    if train:
        while not created_dir:
            try:
                os.mkdir(models)
                created_dir = True
            except OSError as e:
                created_dir = False
                n += 1
                models = os.path.join(args.timestp,'models_{}'.format(n))
                if n > 100:
                    raise e
        
        print('STARTING TRAINING')
        # Callbacks:
        tb = keras.callbacks.TensorBoard(log_dir=os.path.join(models,'logs4'), histogram_freq=0, write_graph=True, write_images=True)
        es = keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=20,restore_best_weights=True)
        lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)
        try:
            vae.fit(my_training_batch_generator,
                    steps_per_epoch=(num_train // batch_size),
                    epochs=200,
                    #initial_epoch=118,
                    #verbose=1,
                    validation_data=my_validation_batch_generator,
                    validation_steps=((num_samples-num_train) // batch_size),
                    callbacks=[tb,es])
        except Exception as e:
            logger.error(e)
            return

        
        encoder.save(os.path.join(models,"VAE_encoder")) 
        decoder.save(os.path.join(models,"VAE_decoder"))
        #vae.save(os.path.join(models,"VAE.h5"))
        encoder.save_weights(os.path.join(models,'encoder_weights.h5'))
        decoder.save_weights(os.path.join(models,'decoder_weights.h5'))
    else:
        #encoder =  keras.models.load_model(os.path.join(models,'VAE_encoder'))
        #decoder =  keras.models.load_model(os.path.join(models,'VAE_decoder'))
        #vae = VAE(encoder, decoder)
        #vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005))
        encoder.load_weights(os.path.join(models,'encoder_weights5.1.h5'))
        decoder.load_weights(os.path.join(models,'decoder_weights5.1.h5'))
        #encoder.save_weights(os.path.join(models,'encoder_weights.h5'))
        #decoder.save_weights(os.path.join(models,'decoder_weights.h5'))
        
        x_test = my_validation_batch_generator.__getitem__(0)
        print(x_test.shape)
        
        encoded_data = encoder.predict(x_test)
        decoded_data = decoder.predict(encoded_data[2])

        for n in range(100):
            img = decoded_data[n,:,:,:]
            org = x_test[n,:,:]
            img[img < 0] = 0
            #img = abs(img)
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
