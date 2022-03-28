import cv2 as cv
import os
import numpy as np


def generate_data(num_samples):
    path = 'C:\\Users\\emiwin\\exjobb\\ellipses2'
    try:
        os.mkdir(path)
    except OSError:
        print('whoops')

    ang = np.linspace(-2,2,9)
    ax = np.ones(9) + abs(np.linspace(-0.3,0.3,9))
    c = np.linspace(2,-2,9) #200,200


    for i in range(num_samples):
        ax1 = np.random.uniform(10,100)
        ax2 = np.random.uniform(10,100)
        angle = np.random.uniform(-30,30)
        c1 = np.random.uniform(225,275) # 100,412
        c2 = np.random.uniform(225,275)

        for n in range(9):
            name = os.path.join(path,'ellips_'  + str(i).zfill(5) + '_' + str(n).zfill(2) + '.png')
            img = np.zeros(shape=(512,512), dtype=np.uint8)

            img = cv.ellipse(img,center=(int(c1+c[n]),int(c2)),axes=(int(ax1*ax[n]),int(ax2*ax[n])),angle=int(angle*ang[n]),
                                startAngle=180,endAngle=-180,color=235,thickness=-1)

            cv.imwrite(name,img)


def main():
    img = np.zeros(shape=(512,512), dtype=np.uint16)
    #image , center Coordinates , axes Length , angle , start Angle , end Angle , color [, thickness [, line Type [, shift ] ] ] )
    angle = 0
    startAngle=180
    endAngle=-180
    ax1 = 0
    ax2 = 0

    generate_data(5)
    while False:
        img = np.zeros(shape=(512,512), dtype=np.uint16)
        img = cv.ellipse(img,center=(256,256),axes=(ax1,ax2),angle=angle,
                            startAngle=startAngle,endAngle=endAngle,color=65535,thickness=2)
        cv.imshow('Ellips',img)
        k=cv.waitKey(1)
        if k & 0xFF == ord('q'):
            break
        elif  k & 0xFF == ord('a'):
            angle += 1
            print('angle=',angle)
        elif  k & 0xFF == ord('s'):
            angle -= 1
            print('angle=',angle)
        elif  k & 0xFF == ord('d'):
            ax1 += 1
            print('ax1=',ax1)
        elif  k & 0xFF == ord('f'):
            ax1 -= 1
            print('ax1=',ax1)
        elif  k & 0xFF == ord('g'):
            ax2 += 1
            print('ax1=',ax2)
        elif  k & 0xFF == ord('h'):
            ax2 -= 1
            print('ax1=',ax2)

    


if __name__ == '__main__':
    main()