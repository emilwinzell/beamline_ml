import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



def main():
    image = np.ones((300,300),dtype = np.uint8)
    image[:100,:100] = 125
    image[:100,100:200] = 170
    image[200:,200:] = 225
    plt_image = np.float32(image)/image.max()
    cv.imshow('tesst',image)
    cv.waitKey(0)
    imgplot= plt.imshow(plt_image)
    plt.show()


if __name__ == '__main__':
    main()