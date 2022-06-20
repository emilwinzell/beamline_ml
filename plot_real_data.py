
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import xml.etree.ElementTree as ET
import argparse
import glob
import pandas as pd
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--base", default="C:\\Users\\emiwin\\exjobb\\real_data", help="path to directory")
    args = parser.parse_args()
    
    df_i = pd.read_csv(os.path.join(args.base,'4437_intensity.csv'))
    df_x = pd.read_csv(os.path.join(args.base,'4437_x.csv'))
    df_y = pd.read_csv(os.path.join(args.base,'4437_y.csv'))

    # df_i = pd.read_csv(os.path.join(args.base,'4436_a_intensity.csv'))
    # df_x = pd.read_csv(os.path.join(args.base,'4436_a_mp1_x.csv'))
    # df_y = pd.read_csv(os.path.join(args.base,'4436_a_mp1_y.csv'))

    itsy = df_i.to_numpy()
    x = df_x.to_numpy()
    y = df_y.to_numpy()

    itsy = np.squeeze(itsy)
    x = np.squeeze(x)
    y = np.squeeze(y)
    
    #img = np.reshape(itsy[:784],(28,28))
    #n_img = img/np.max(img)*255
    #cv.imshow('img',n_img.astype(np.uint8))
    #cv.waitKey(0)

    plt.figure()
    plt.plot(itsy)
    plt.title('intensity')

    plt.figure()
    plt.plot(x)
    plt.title('x')

    plt.figure()
    plt.plot(y)
    plt.title('y')

    plt.figure()
    plt.tricontour(x,y,itsy)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend('Intensity')
    plt.show()



if __name__ == '__main__':
    main()