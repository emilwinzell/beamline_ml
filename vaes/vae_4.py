# vae_4
#
# Trying differnet implementation
# from: https://keras.io/examples/generative/vae/?msclkid=0a8e33e0a5ff11ecab29a7e68ba38092
#
# New architecture from: https://github.com/IsaacGuan/3D-VAE/blob/master/VAE.py
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
from tensorflow.keras.activations import sigmoid
import glob

import logging
logging.basicConfig(filename='./vae_3.log', level=logging.DEBUG)
logger=logging.getLogger(__name__)


class My_Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.x, self.y = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size) / 9))

    def __normalize__(self,img,bytes=8):
        m = float(2**bytes-1)
        if img.max() > 0:
            norm_img = img.astype(np.float32)
            norm_img = norm_img/m
            return 3.0*norm_img-1.0 # from paper...
        else:
            return 3.0*img.astype(np.float32)-1.0

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size*9:(idx + 1) * self.batch_size*9]
        batch_y = self.y[idx * self.batch_size*9:(idx + 1) * self.batch_size*9]

        targets = []
        img3d = []
        b=8
        for img_name in batch_x:
            img = cv.imread(img_name,-1)
            if str(img.dtype)=='uint16':
                b=16 
            img = cv.blur(img,(5,5))
            img = cv.resize(img,(256,256))
            img = self.__normalize__(img,b)
            img3d.append(img)
            if len(img3d) == 9:
                targets.append(img3d)
                img3d = []
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
        loss = -(98.0 * target * K.log(output) + 2.0 * (1.0 - target) * K.log(1.0 - output)) / 100.0
        return loss

    def __loss_fcn(self,data):
        
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction = tf.squeeze(reconstruction)
        kl_div = -0.5 * K.mean(1 + 2 * z_log_var - K.square(z_mean) - K.exp(2 * z_log_var))
        voxel_loss = K.cast(K.mean(self.weighted_binary_crossentropy(data, K.clip(sigmoid(reconstruction), 1e-7, 1.0 - 1e-7))), 'float32')

        # sums up across depth, width and height, then mean over batch 
        #reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data-reconstruction),axis=(1,2,3)))
        #kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        #kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = kl_div+voxel_loss#reconstruction_loss + kl_loss
        return voxel_loss, kl_div,total_loss


    def train_step(self, data):
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


def build_encoder(width=256,height=256,depth=9,latent_space_dim=5):
    print('ENCODER')
    encoder_input = keras.layers.Input(shape=(depth,width,height,1), name='encoder_input')
    
    
    x = keras.layers.BatchNormalization()(
        keras.layers.Conv3D(
            filters = 8,
            kernel_size = (3, 3, 3),
            strides = (1, 1, 1),
            padding = 'valid',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'encoder_conv1')(encoder_input))
    print(encoder_input.shape)
    print(x.shape)

    x = keras.layers.BatchNormalization()(
        keras.layers.Conv3D(
            filters = 8,
            kernel_size = (3, 3, 3),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'encoder_conv2')(encoder_input))
    print(x.shape)
    
    x = keras.layers.BatchNormalization()(
        keras.layers.Conv3D(
            filters = 16,
            kernel_size = (3, 3, 3),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'encoder_conv3')(x))
    print(x.shape)

    x = keras.layers.BatchNormalization()(
        keras.layers.Conv3D(
            filters = 16,
            kernel_size = (3, 3, 3),
            strides = (1, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'encoder_conv4')(x))
    print(x.shape)

    x = keras.layers.BatchNormalization()(
        keras.layers.Conv3D(
            filters = 16,
            kernel_size = (3, 3, 3),
            strides = (1, 1, 1),
            padding = 'valid',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'encoder_conv5')(x))
    print(x.shape)

    x = keras.layers.BatchNormalization()(
        keras.layers.Conv3D(
            filters = 32,
            kernel_size = (3, 3, 3),
            strides = (1, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'encoder_conv6')(x))
    print(x.shape)

    x = keras.layers.BatchNormalization()(
        keras.layers.Conv3D(
            filters = 32,
            kernel_size = (3, 3, 3),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'encoder_conv7')(x))
    print(x.shape)

    shape_before_flatten = keras.backend.int_shape(x)[1:]
    print(np.prod(shape_before_flatten))
    enc_flatten = keras.layers.Flatten()(x)
    x = keras.layers.BatchNormalization()(
        keras.layers.Dense(
            units = np.prod(shape_before_flatten),
            kernel_initializer = 'glorot_normal',
            activation = 'elu')(enc_flatten))

    encoder_mu = keras.layers.Dense(units=latent_space_dim, name="encoder_mu")(x)
    encoder_log_variance = keras.layers.Dense(units=latent_space_dim, name="encoder_log_variance")(x)

    z = Sampling()([encoder_mu,encoder_log_variance])

    model = keras.models.Model(encoder_input, [encoder_mu, encoder_log_variance, z], name="encoder_model")

    return model,shape_before_flatten


def build_decoder(shape,latent_space_dim=5):
    print('DECODER')
    decoder_input = keras.layers.Input(shape=(latent_space_dim,), name="decoder_input")

    x = keras.layers.BatchNormalization()(
        keras.layers.Dense(
            units = np.prod(shape),
            kernel_initializer = 'glorot_normal',
            activation = 'elu')(decoder_input))
    x = keras.layers.Reshape(
        target_shape = shape)(x)
    print(x.shape)

    x = keras.layers.BatchNormalization()(
        keras.layers.Conv3DTranspose(
            filters = 32,
            kernel_size = (3, 3, 3),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'decoder_conv1')(x))
    print(x.shape)

    x = keras.layers.BatchNormalization()(
        keras.layers.Conv3DTranspose(
            filters = 32,
            kernel_size = (3, 3, 3),
            strides = (1, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'decoder_conv2')(x))
    print(x.shape)

    x = keras.layers.BatchNormalization()(
        keras.layers.Conv3DTranspose(
            filters = 16,
            kernel_size = (3, 3, 3),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'decoder_conv3')(x))
    print(x.shape)

    x = keras.layers.BatchNormalization()(
        keras.layers.Conv3DTranspose(
            filters = 16,
            kernel_size = (3, 3, 3),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'decoder_conv4')(x))
    print(x.shape)

    x = keras.layers.BatchNormalization()(
        keras.layers.Conv3DTranspose(
            filters = 16,
            kernel_size = (3, 3, 3),
            strides = (2, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'decoder_conv5')(x))
    print(x.shape)


    x = keras.layers.BatchNormalization()(
        keras.layers.Conv3DTranspose(
            filters = 8,
            kernel_size = (2, 1, 1),
            strides = (1, 1, 1),
            padding = 'valid',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'decoder_conv6')(x))
    print(x.shape)

    x = keras.layers.BatchNormalization()(
        keras.layers.Conv3DTranspose(
           filters = 8,
            kernel_size = (4, 4, 4),
            strides = (1, 2, 2),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'decoder_conv7')(x))
    print(x.shape)


    decoder_output = keras.layers.BatchNormalization(
        beta_regularizer = keras.regularizers.l2(0.001),
        gamma_regularizer = keras.regularizers.l2(0.001))(
        keras.layers.Conv3DTranspose(
            filters = 1,
            kernel_size = (3, 3, 3),
            strides = (1, 1, 1),
            padding = 'same',
            kernel_initializer = 'glorot_normal',
            activation = 'elu',
            name = 'decoder_conv8')(x))
    print(decoder_output.shape)
    
    model = keras.models.Model(decoder_input, decoder_output, name="decoder_model")
    return model

def learning_rate_scheduler(epoch, lr):
    if epoch >= 1:
        lr = 0.005
    return lr

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
    img_size = (256,256)
    depth = 9
    encoder, shape = build_encoder(width=img_size[0],height=img_size[1],depth=depth,latent_space_dim=latent_space_dim)
    #encoder.summary()

    decoder = build_decoder(shape,latent_space_dim=latent_space_dim)
    #decoder.summary()
    #keras.utils.plot_model(encoder, to_file = 'vae_encoder.pdf', show_shapes = True)
    #keras.utils.plot_model(decoder, to_file = 'vae_decoder.pdf', show_shapes = True)


    vae = VAE(encoder, decoder)

    sgd = keras.optimizers.SGD(lr = 0.0001, momentum = 0.9, nesterov = True)
    vae.compile(optimizer=sgd)#,run_eagerly=True)
    

    models = os.path.join(args.timestp,'models_1')   
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
        #es = keras.callbacks.EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True)
        lrs = keras.callbacks.LearningRateScheduler(learning_rate_scheduler)
        try:
            vae.fit(my_training_batch_generator,
                    steps_per_epoch=(num_train // batch_size),
                    epochs=10,
                    validation_data=my_validation_batch_generator,
                    validation_steps=((num_samples-num_train) // batch_size),
                    callbacks=[tb,lrs])
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
        encoder.load_weights(os.path.join(models,'encoder_weights.h5'))
        decoder.load_weights(os.path.join(models,'decoder_weights.h5'))
        #encoder.save_weights(os.path.join(models,'encoder_weights.h5'))
        #decoder.save_weights(os.path.join(models,'decoder_weights.h5'))
        
        x_test = my_validation_batch_generator.__getitem__(0)
        print(x_test.shape)
        
        encoded_data = encoder.predict(x_test)
        decoded_data = decoder.predict(encoded_data[2])

        for n in range(10):
            for i in range(9):
                img = decoded_data[n,i,:,:,:]
                org = x_test[n,i,:,:]
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
