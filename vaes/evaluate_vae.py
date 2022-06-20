"""
Program to evaluate the VAE on the test set, calculate results and present in plots

Emil Winzell, May 2022

"""


import sys
#sys.stdout = open('output.txt','wt')
import os
import cv2 as cv
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from tensorflow import keras
import glob
import random
from scipy import optimize
from matplotlib import pyplot as plt


from vae2D_4 import My_Generator, VAE, Sampling



def build_encoder(width=256,height=256,latent_space_dim=10):
    print('ENCODER')
    encoder_input = keras.layers.Input(shape=(width,height,1), name='encoder_input')
    ex_input = keras.layers.Input(shape=(4,), name='extra_input')
    y = keras.layers.Dense(16,activation='relu')(ex_input)
    y = keras.layers.Dense(8,activation='relu')(y)

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

    y = keras.layers.Dense(16,name="decoder_dense_ex_1")(decoder_input)
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


def evaluate(img,scale,plot=False):
    x1D = np.sum(img, axis=0)
    y1D = np.sum(img, axis=1)
    cx,dx,cy,dy = scale

    
    xbins = np.linspace(cx-dx/2,cx+dx/2,len(x1D))
    ybins = np.linspace(cy-dy/2,cy+dy/2,len(y1D))

    # PLOT HISTOGRAMS WITH TRUE AND RECONSTRUCTED IMAGES
    # if plot == False:
    #     plt.figure()
    # sub1 = plt.subplot(121)
    # plt.plot(xbins,x1D)
    # sub1.set_title('x-axis')
    # sub1.set_ylabel('Counts')
    # sub1.set_xlabel('Bins')
    # #plt.plot(x,g_xt)
    # sub2 = plt.subplot(122)
    # plt.plot(ybins,y1D)
    # sub2.set_title('y-axis')
    # sub2.set_xlabel('Bins')
    # if plot == True:
    #     sub2.margins(x=0.05, y=0.15)
    #     plt.legend(['True (encoder input)','Predicted (decoder output)'], loc='upper left')
    #     plt.show()

    fwhmX = calculate_fwhm(x1D,xbins)
    fwhmY = calculate_fwhm(y1D,ybins)
    return fwhmX, fwhmY


def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 256
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([np.append([xi,yi],np.random.uniform(0,1,8))])
            z_sample = np.array([np.append([xi,yi],np.zeros(8))])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            print('i: {},j: {}, scale={}'.format(i,j,x_decoded[1]))
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",default='/home/emiwin/exjobb/' ,help="path to model")
    parser.add_argument("-t", "--timestp", default='/home/emiwin/exjobb/', help="path to timestamp folder")
    args = parser.parse_args()

    # BUILD MODEL AS IT WAS TRAINED
    latent_space_dim = 10
    img_size = (256,256)
    encoder, shape = build_encoder(width=img_size[0],height=img_size[1],latent_space_dim=latent_space_dim)
    encoder.summary()

    decoder = build_decoder(shape,latent_space_dim=latent_space_dim)
    decoder.summary()

    histograms = os.path.join(args.timestp,'histograms')
    images = os.path.join(args.timestp,'images')
    xml = os.path.join(args.timestp,'data.xml')
    tree = ET.parse(xml)
    root = tree.getroot()
    
    test_imgs = glob.glob(os.path.join(images,'*.png'))
    num_samples = len(test_imgs)
    
    #random.shuffle(test_imgs)
    print('loaded {} samples'.format(num_samples))
    
    # LOAD THE WEIGHTS OF THE MODEL TO BE EVALUATED
    encoder.load_weights(os.path.join(args.model,'encoder_weights.h5')) 
    decoder.load_weights(os.path.join(args.model,'decoder_weights.h5'))

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),run_eagerly=False)

    # Batch generators:
    batch_size = num_samples//5
    my_test_batch_generator = My_Generator(test_imgs, root, batch_size)
    
    # FOR SAVING THE RECONSTRUCTED IMAGES
    # dec_imgs = os.path.join(args.timestp,'decoded_images')
    # try:
    #     os.mkdir(dec_imgs)
    # except OSError as e:
    #     print('decoded exists!')

    # PLOT VISUAL OF THE LATENT SPACE
    #plot_latent_space(vae,n=10)
    samp_nr = 0
    X = []
    Y = []
    small_sc = []
    for i in range(num_samples//batch_size):
        x_test = my_test_batch_generator.__getitem__(i)
        
        input_imgs = x_test[0]
        input_scales = x_test[1]

        encoded_data = encoder.predict(x_test)
        [output_imgs,output_scales] = decoder.predict(encoded_data[2])
        
        xdiff=[]
        ydiff=[]
        scaling = []
        
        sys.stdout = sys.__stdout__
        for i_img,i_sc,o_img,o_sc in zip(input_imgs,input_scales,output_imgs,output_scales):
            tx,ty = evaluate(i_img,i_sc) # get true FWHM values
            o_img = np.squeeze(o_img)
            o_img[o_img < 0] = 0
            img1 = o_img*255
            img1 = img1.astype(np.uint8)
            #img1 = cv.blur(img1, (20,20))
            img2 = i_img*255
            img2 = img2.astype(np.uint8)

            # SAVING THE RECONSTRUCTED IMAGES
            #save_name = os.path.join(dec_imgs,'dec_{}.png'.format(samp_nr))
            #cv.imwrite(save_name,img1)
            samp_nr += 1

            px,py = evaluate(o_img,o_sc,True) # get reconstructed FWHM values
            sc_err = abs(np.array(o_sc)-np.array(i_sc))
            if i_sc[1] < 0.01:
                small_sc.append(sc_err)
            
            # COMPENSATION FOR STATIONARY ERROR (see report)
            #px = px - 0.3527/(1+0.3527)*px
            #py = py -  1.015/(1+1.015)*py

            xdiff.append(abs(tx-px)/tx)
            ydiff.append(abs(ty-py)/ty)
            scaling.append(sc_err)

        scaling = np.array(scaling)
        print('Average batch FWHM errors: lateral={}, vertical={}'.format(np.mean(xdiff),np.mean(ydiff)))
        print('Average batch scaling error: ', np.mean(scaling, axis=0))
        print('Batch scaling variance: ', np.var(scaling, axis=0))
        X = X + xdiff
        Y = Y + ydiff


    print('Average small scaling error: ', np.mean(small_sc, axis=0))
    plt.figure()
    plt.hist(X,bins=100,range=(0.0, 2.5))
    plt.axvline(np.mean(X), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(np.mean(X)*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(X)))
    plt.xlabel('FWHM error, fraction of true value')
    plt.ylabel('Counts')
    plt.title('Lateral (x-axis)')

    plt.figure()
    plt.hist(Y,bins=100,range=(0.0, 2.5))
    plt.axvline(np.mean(Y), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(np.mean(Y)*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(Y)))
    plt.xlabel('FWHM error, fraction of true value')
    plt.ylabel('Counts')
    plt.title('Vertical (y-axis)')
    plt.show()




if __name__ == '__main__':
    main()
