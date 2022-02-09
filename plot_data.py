import os
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import argparse



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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--base", help="path to base directory (timestamp)")
    args = parser.parse_args()
    images = os.path.join(args.base,'images')
    labels = os.path.join(args.base,'labels.csv')

    # The attributes for creating the data
    pitch = np.linspace(1,5,5)
    yaw = np.array([0, 0.01, 0.05])
    roll = np.array([0, 0.01])

    data_df = pd.read_csv(labels)
    plot_pitch(images,data_df,pitch,yaw,roll)

if __name__ == '__main__':
    main()
