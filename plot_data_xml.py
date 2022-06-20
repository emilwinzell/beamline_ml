"""
Plot some data (histograms and fwhms)
ex:
python .\plot_data_xml.py -p 'C:\Users\emiwin\exjobb\04061435'  

Emil Winzell, April 2022

"""



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


def calculate_fwhm(data,lim,N):
        x = np.linspace(lim[0], lim[-1], N)
        n = np.count_nonzero(data==0)//2+1
        est_std = (lim[-n]-lim[n])/5.0
        est_mean = lim[len(lim)//2]
        est_amp = np.max(data)*2.5
        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-((x - mean) / np.sqrt(2) / stddev)**2)
        
        [amp,mean,std],_ = optimize.curve_fit(gaussian, lim, data,p0=[est_amp,est_mean,est_std])
        std = abs(std)
        fwhm = 2*np.sqrt(2*np.log(2))*std

        return fwhm

def fit_poly(data,xlim,N):
    x = np.linspace(xlim[0], xlim[-1], N)

    def poly(x, a0,a1,a2):
        return a0 + a1 *x + a2*x**2

    [a0,a1,a2],_ = optimize.curve_fit(poly, xlim, data)
    print(a0,a1,a2)
    
    if a2 < 0:
        amin = -a1/(2*a2)
    else:
        amin = -a1/(2*a2)
        #amin = x[np.argmin(poly(x, a0,a1,a2))]
    print(amin)
    return poly(x,a0,a1,a2),x,amin,poly(amin,a0,a1,a2)

def fit_linear(data,xlim,N):
    x = np.linspace(xlim[0], xlim[-1], N)

    ub = np.max(data)
    lb = np.min(data)
    mapk = 28/(ub-lb)
    mapm = 14-mapk*ub

    rescaled = mapk*np.array(data)+mapm

    def dis_cont(x, a1,b1,a2,b2,c):
        return np.where(x <= c, a1*x+b1, a2*x+b2)

    [a1,b1,a2,b2,c],_ = optimize.curve_fit(dis_cont, xlim, data)
    [k1,m1,k2,m2,r],_ = optimize.curve_fit(dis_cont, xlim, rescaled)
    
    

    inc1 = np.arctan(k1)*180/np.pi
    inc2 = np.arctan(k2)*180/np.pi
    #print(inc1,inc2)
    value = abs(inc1+inc2)
    return dis_cont(x, a1,b1,a2,b2,lb),value

def plot_histograms(path):
    list_of_hist = glob.glob(path + '\\*.csv')
    n = 1
    xFWHMs = []
    yFWHMs = []
    dqs = np.linspace(-14, 14, 9)
    for hist in list_of_hist:
        #print(hist)
        end = os.path.split(hist)[-1]
        snr = int(end.split('_')[1])
        #if not snr == 98:
        #    continue 
        df = pd.read_csv(hist)
        xt1D = df["xt1D"].to_numpy()
        xBins = df["xbinEdges"].to_numpy()
        yt1D = df["zt1D"].to_numpy()
        yBins = df["zbinEdges"].to_numpy()

        try:
            x_fwhm = calculate_fwhm(xt1D,xBins,1000)
            y_fwhm = calculate_fwhm(yt1D,yBins,1000)
            plt.figure()
            sub1 = plt.subplot(121)
            plt.plot(xBins,xt1D)
            sub1.set_title('x-axis')
            sub1.set_ylabel('Counts')
            sub1.set_xlabel('Bins')
            #plt.plot(x,g_xt)
            sub2 = plt.subplot(122)
            plt.plot(yBins,yt1D)
            sub2.set_title('y-axis')
            sub2.set_xlabel('Bins')
            #plt.plot(y,g_yt)
        except RuntimeError:
            plt.figure()
            plt.plot(yBins,yt1D)
            plt.show()
            return
        xFWHMs.append(x_fwhm)
        yFWHMs.append(y_fwhm)
        if n==9:
    
            plt.figure()
            plt.rcParams.update({'font.size': 14})
            plt.scatter(dqs,xFWHMs)
            
            p_xt,x,dx0,minX = fit_poly(xFWHMs,np.linspace(-14,14,9),1000)
            y_lin,v = fit_linear(yFWHMs,np.linspace(-14,14,9),1000)
            dy0 = dqs[np.argmin(yFWHMs)]
            plt.scatter(dx0,minX)
            #print(dy0, v)
            #plt.figure()
            plt.plot(x,p_xt)
            plt.xlabel('z-axis (mm)')
            plt.ylabel('FWHM (mm)')
            plt.figure()
            plt.scatter(dqs,yFWHMs)
            plt.scatter(dy0,min(yFWHMs))
            plt.xlabel('z-axis (mm)')
            plt.ylabel('FWHM (mm)')

            #gap = abs(x[np.argmin(p_xt)]-y[np.argmin(p_yt)])
            gap = abs(dx0-dy0)
            FWHMx = min(xFWHMs)
            FWHMy = min(yFWHMs)
            print('gap: ', gap)
            print('fx: ', FWHMx)
            print('fy: ', FWHMy)
            print('vad: ', v)
            
            #plt.figure()
            plt.plot(x,y_lin)
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
        blurred = cv.blur(img,(5,5))
        small = cv.resize(blurred,(256,256))
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
    #labels = os.path.join(args.base,'data.xml')

    #tree = ET.parse(labels)
    #root = tree.getroot()
    """
    pitches = np.linspace(-10,10,5)*1e-4
    yaws = np.linspace(-0.03,0.03,5)
    rolls = np.linspace(-0.03,0.03,5)
    transl = np.linspace(-1,1,3) 
    """
    #c = input('choose attrib (p,y,r,x or y): ')
    #imgnr = input('choose img number: ')
    
    
    #img_blur(images)
    plot_histograms(histograms)


if __name__ == '__main__':
    main()