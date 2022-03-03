import sys
import os
import cv2 as cv
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import glob
import math


def step_fcn(x):
    if x < 0:
        return 0
    else:
        return x

def resize_img(img,x,y,dx,dy):
    pixels = 1000
    limits = [-1,1] #mm
    pix_per_mm = pixels/float(limits[1]-limits[0])
    

    # resize image to correct number of pixels
    dsize = (int((dx-x)*pix_per_mm), int((dy-y)*pix_per_mm))
    r_img = cv.resize(img,dsize)
    # pad image to correct dimensions
    top = math.ceil((y-limits[0])*pix_per_mm)
    bottom = math.ceil((limits[1]-dy)*pix_per_mm)
    left = math.ceil((x-limits[0])*pix_per_mm)
    right = math.ceil((limits[1]-dx)*pix_per_mm)

    pad_img = cv.copyMakeBorder(r_img,step_fcn(top),step_fcn(bottom),
                                step_fcn(left),step_fcn(right),
                                borderType=cv.BORDER_CONSTANT,value=[0,0,0])
    h,w,_ = np.shape(pad_img)

    pad_img = pad_img[step_fcn(-1*top):h-step_fcn(-1*bottom),step_fcn(-1*left):w-step_fcn(-1*right),:]
    pad_img = pad_img[0:pixels,0:pixels,:]
    return pad_img


def get_data(imgnr,root,base_name,images):
    targets = []
    paths = []
    for item in root:
        if item.tag[:6] == 'sample':
            samplenr = item.tag[7:]
            for subitem in item:
                if subitem.tag == 'images':
                    imgname =  base_name + '_'  + samplenr.zfill(5) + '_' + str(imgnr).zfill(2) + '.png'
                    paths.append(os.path.join(images,imgname))
                    #img = cv.imread(img_path,0).astype(np.float32)
                    #img.flatten()
                    for i in subitem:
                        if i.attrib['file'] == imgname:
                            cx = float(i.attrib['centerx'])
                            dx =  float(i.attrib['dx'])
                            cz = float(i.attrib['centerz'])
                            dz = float(i.attrib['dz'])
                            img = (cx-dx/2,cz-dz/2,cx+dx/2,cz+dz/2)
                            
                    targets.append(img)
    targets = np.array(targets)

    return targets,paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--base", help="path to base directory (timestamp)")
    args = parser.parse_args()
    images = os.path.join(args.base,'images')
    histograms = os.path.join(args.base,'histograms')
    labels = os.path.join(args.base,'data.xml')
    base_name = os.path.split(args.base)[1]

    tree = ET.parse(labels)
    root = tree.getroot()
    
    targets,paths = get_data(4,root,base_name,images)
    
    for i,(x,y,dx,dy) in enumerate(targets):
        #print(x,y,dx,dy)
        img = cv.imread(paths[i])
        pad_img = resize_img(img,x,y,dx,dy)
        print(np.shape(pad_img))
        cv.imshow('padded',pad_img)
        cv.waitKey(0)
   
   


if __name__ == '__main__':
    main()