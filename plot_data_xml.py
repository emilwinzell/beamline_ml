import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import xml.etree.ElementTree as ET
import argparse
import glob
import pandas as pd

from scipy import optimize


def plot_pitch(img_path,df,pitch,yaw,roll):
    fig=plt.figure()
    rows = len(yaw)*len(roll)
    cols = len(pitch)

    i = 1
    for y in yaw:
        for r in roll:
            for p in pitch:
                img_names = list(df['img'].loc[(df['pitch'] == p) & (df['yaw'] == y) & (df['roll'] == r)])
                img = cv.imread(os.path.join(img_path,img_names[0]))
                #img = np.array(img, dtype=np.float32)
                fig.add_subplot(rows, cols, i)
                plt.imshow(img)
                plt.title('P={0},Y={1},R={2}'.format(p,y,r),fontsize=5)
                plt.axis('off')
                i += 1

    c_head = ['Pitch = {}'.format(c) for c in pitch]

    plt.tight_layout()
    plt.show()


def fit_gaussian(data,xlim,N):
    x = np.linspace(xlim[0], xlim[-1], N)

    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) / np.sqrt(2) / stddev)**2)
    
    [amp,mean,std],_ = optimize.curve_fit(gaussian, xlim, data)
    std = abs(std)

    fwhm = 2*np.sqrt(2*np.log(2))*std


    return gaussian(x,amp,mean,std),x,fwhm,mean




def plot_histograms(path):
    list_of_hist = glob.glob(path + '\\*.csv')
    n = 1
    xFWHMs = []
    yFWHMs = []
    dqs = np.linspace(-14, 14, 9)
    for hist in list_of_hist:
        df = pd.read_csv(hist)
        xt1D = df["xt1D"].to_numpy()
        xBins = df["xbinEdges"].to_numpy()
        yt1D = df["zt1D"].to_numpy()
        yBins = df["zbinEdges"].to_numpy()

        g_xt,x,x_fwhm,xmu = fit_gaussian(xt1D,xBins,1000)
        g_yt,y,y_fwhm,ymu = fit_gaussian(yt1D,yBins,1000)
        xFWHMs.append(x_fwhm)
        yFWHMs.append(y_fwhm)
        if n==9:
            plt.figure()
            plt.scatter(dqs,xFWHMs)
            plt.figure()
            plt.scatter(dqs,yFWHMs)
            plt.show()
            n=1
            xFWHMs = []
            yFWHMs = []
        else:
            n+=1


        # print(x_fwhm)
        # print(z_fwhm)

        # plt.subplot(121)
        # plt.plot(xBins,xt1D)
        # plt.plot(x,g_xt)
        # plt.plot([xm-x_fwhm/2,xm+x_fwhm/2],[np.max(xt1D)/2,np.max(xt1D)/2])
        # plt.title('x axis')
        # plt.subplot(122)
        # plt.plot(zBins,zt1D)
        # plt.plot(z,g_zt)
        # plt.title('y(z) axis')
        # plt.show()
        #if input('continue?')=='n':
        #    break


def img_blur(path):
    list_of_images = glob.glob(path + '\\*.png')
    for img_path in list_of_images:
        img = cv.imread(img_path,-1)
        if str(img.dtype) == 'uint16':
            print('lololol')
        print(np.max(img))
        blurred = cv.blur(img,(5,5))
        small = cv.resize(blurred,(256,256))
        print(np.max(small))
        cv.imshow('org',img)
        cv.imshow('blurred',blurred)
        cv.imshow('small',small)
        cv.waitKey(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--base", help="path to base directory (timestamp)")
    args = parser.parse_args()
    images = os.path.join(args.base,'images')
    histograms = os.path.join(args.base,'histograms')
    labels = os.path.join(args.base,'data.xml')

    tree = ET.parse(labels)
    root = tree.getroot()
    """
    pitches = np.linspace(-10,10,5)*1e-4
    yaws = np.linspace(-0.03,0.03,5)
    rolls = np.linspace(-0.03,0.03,5)
    transl = np.linspace(-1,1,3) 
    """
    #c = input('choose attrib (p,y,r,x or y): ')
    #imgnr = input('choose img number: ')
    
    
    img_blur(images)
    plot_histograms(histograms)


    



if __name__ == '__main__':
    main()