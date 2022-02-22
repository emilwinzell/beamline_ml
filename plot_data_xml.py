import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import xml.etree.ElementTree as ET
import argparse
import glob
import pandas as pd


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

def plot_histograms(path):
    list_of_hist = glob.glob(path + '\\*.csv')
    for hist in list_of_hist:
        df = pd.read_csv(hist)
        xt1D = df["xt1D"].to_numpy()
        xBins = df["xbinEdges"].to_numpy()
        plt.plot(xBins,xt1D)
        plt.show()
        if input('continue?')=='n':
            break



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

    c = input('choose attrib (p,y,r,x or y):')
    """

    plot_histograms(histograms)


    



if __name__ == '__main__':
    main()