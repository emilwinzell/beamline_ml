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

from vae_4 import build_encoder, build_decoder

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


def weighted_binary_crossentropy(target, output):
        loss = -(98.0 * target * K.log(output) + 2.0 * (1.0 - target) * K.log(1.0 - output)) / 100.0
        return loss


def learning_rate_scheduler(epoch, lr):
    if epoch >= 1:
        lr = 0.005
    return lr

def main():

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
    enc_mod = build_encoder(width=img_size[0],height=img_size[1],depth=depth,latent_space_dim=latent_space_dim)
    encoder = enc_mod['encoder']
    enc_input = enc_mod['inputs']
    mu = enc_mod['mu']
    sigma = enc_mod['sigma']
    #encoder.summary()

    decoder = build_decoder(enc_mod['shape'],latent_space_dim=latent_space_dim)
    dec_output = decoder(encoder(enc_input)[2])
    #decoder.summary()
    #keras.utils.plot_model(encoder, to_file = 'vae_encoder.pdf', show_shapes = True)
    #keras.utils.plot_model(decoder, to_file = 'vae_decoder.pdf', show_shapes = True)


    vae = keras.models.Model(enc_input, dec_output)
    kl_div = -0.5 * K.mean(1 + 2 * sigma - K.square(mu) - K.exp(2 * sigma))
    voxel_loss = K.cast(K.mean(weighted_binary_crossentropy(enc_input, K.clip(sigmoid(dec_output), 1e-7, 1.0 - 1e-7))), 'float32')  + kl_div
    vae.add_loss(voxel_loss)

    sgd = keras.optimizers.SGD(lr = 0.0001, momentum = 0.9, nesterov = True)
    vae.compile(optimizer=sgd, metrics = ['accuracy'])#,run_eagerly=True)
    

    models = os.path.join(args.timestp,'models_1')   
    created_dir = False
    n = 1

    # Batch generators:
    batch_size = 10
    my_training_batch_generator = My_Generator(x_train, y_train, batch_size)
    my_validation_batch_generator = My_Generator(x_test, y_test, batch_size)


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
    

if __name__ == '__main__':
    main()
