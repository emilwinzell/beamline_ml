import sys
import os
import cv2 as cv
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from sklearn import neighbors, metrics, cluster
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
    limits = [-2.5,2.5] #mm
    pix_per_mm = pixels/float(limits[1]-limits[0])
    

    # resize image to correct number of pixels
    dsize = (int((dx-x)*pix_per_mm), int((dy-y)*pix_per_mm))
    r_img = cv.resize(img,dsize)
    # pad image to correct dimensions
    top = math.ceil((y-limits[0])*pix_per_mm)+5
    bottom = math.ceil((limits[1]-dy)*pix_per_mm)+5
    left = math.ceil((x-limits[0])*pix_per_mm)+5
    right = math.ceil((limits[1]-dx)*pix_per_mm)+5

    pad_img = cv.copyMakeBorder(r_img,step_fcn(top),step_fcn(bottom),
                                step_fcn(left),step_fcn(right),
                                borderType=cv.BORDER_CONSTANT,value=[0,0,0])
    h,w, = np.shape(pad_img)

    pad_img = pad_img[step_fcn(-1*top):h-step_fcn(-1*bottom),step_fcn(-1*left):w-step_fcn(-1*right)]
    pad_img = pad_img[0:pixels,0:pixels]
    return pad_img

def k_nn(train_features,train_labels,test_features,test_labels):
    # k-NN Classifier
    classifier = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm = 'brute')
    classifier.fit(train_features, train_labels)

    # Predict
    predicted_labels = classifier.predict(test_features)
    dists, nbrs = classifier.kneighbors(test_features)

    # Display results
    print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(test_labels, predicted_labels)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predicted_labels))

    #If you plot the 5 nearest neighbours for the first three classified
    #samples from the test set, you get something similar to this:
    for j in range(3):
        plt.figure(j+1)
        plt.subplot2grid((5,5), (0, 0), rowspan=3, colspan=3)
        image = test_features[j][:-4].reshape(256,256).astype(np.uint8)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title("Label: %s, Predicted: %s" % (test_labels[j],predicted_labels[j]))
        for i in range(5):
            plt.subplot2grid((5,5), (4, i))
            idx = nbrs[j,i]
            image = train_features[idx][:-4].reshape(256,256).astype(np.uint8)
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title("Label: %s" % train_labels[idx])



def k_m_clustering(train_features,train_labels,test_features,test_labels):
    # k-means clustering
    k_means = cluster.KMeans(n_clusters=5)
    clusters = k_means.fit(train_features)

    # Plot the centroids
    plt.figure(4)
    centers = k_means.cluster_centers_
    for i,c in enumerate(centers):
        image = c.reshape(1000,1000).astype(np.uint8)
        plt.subplot(2, 5,i+1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Centroid: %i' % i)

    # Predict
    predicted_labels = k_means.predict(test_features)

    print("Classification report for classifier %s:\n%s\n"
      % (k_means, metrics.classification_report(test_labels, predicted_labels)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predicted_labels))
    
    # Re-map labels
    # predicted_labels_train = k_means.predict(train_features)
    # map_labels = label_clusters(predicted_labels_train,train_labels,5)

    # new_pred_labels = np.zeros((len(predicted_labels),1))
    # for i in range(5):
    #     old = map_labels[i]
    #     new_pred_labels[predicted_labels == old] = i

    # print("New confusion matrix!:\n%s" % metrics.confusion_matrix(test_labels, new_pred_labels))

    # # Completeness: A clustering result satisfies completeness if all the data
    # # points that are members of a given class are elements of the same cluster.
    # print("Completeness score:\n%s" % metrics.completeness_score(train_labels, predicted_labels_train))
    # # Homogenity: A clustering result satisfies homogeneity if all of its
    # # clusters contain only data points which are members of a single class.
    # print("Homogenity score:\n%s" % metrics.homogeneity_score(train_labels, predicted_labels_train))

    plt.show()

def label_clusters(preds,labels,n_clusters):
    cluster_labels = []
    for i in range(n_clusters):
        idx = np.where(labels == i)
        p = preds[idx]
        counts = np.bincount(p)
        cluster_labels.append(np.argmax(counts))
    return cluster_labels



def get_data(imgnr,root,spec,base_name,images):
    org_labels = []
    targets = []
    for item in root:
        if item.tag[:6] == 'sample':
            samplenr = item.tag[7:]
            for subitem in item:
                if subitem.tag == 'specifications':
                    s = subitem.find(spec)
                    org_labels.append(float(s.text))
                elif subitem.tag == 'images':
                    imgname =  base_name + '_'  + samplenr.zfill(5) + '_' + str(imgnr).zfill(2) + '.png'
                    for i in subitem:
                        if i.attrib['file'] == imgname:
                            cx = float(i.attrib['centerx'])
                            dx =  float(i.attrib['dx'])
                            cz = float(i.attrib['centerz'])
                            dz = float(i.attrib['dz'])
                    img_path = os.path.join(images,imgname)
                    img = cv.imread(img_path,0)
                    
                    pad_img=resize_img( img,cx-dx/2,cz-dz/2,cx+dx/2,cz+dz/2)
                    pad_img = pad_img.flatten()
                    targets.append(pad_img)
    targets = np.array(targets)

    uniq = np.unique(org_labels)
    labels = []
    for l in org_labels:
        labels.append(int(np.argwhere(uniq==l)[0]))

    labels = np.array(labels)
    return targets,labels


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
    

    targets,labels = get_data(4,root,'yaw',base_name,images)
   
    print(labels)
    print(np.shape(targets))

    num_examples = np.shape(targets)[0]

    # Split dataset
    num_split = int(0.7*num_examples)
    train_features = targets[:][:num_split]
    train_labels =  labels[:num_split]
    val_features = targets[:][num_split:]
    val_labels = labels[num_split:]

    #k_nn(train_features,train_labels,val_features,val_labels)

    k_m_clustering(train_features,train_labels,val_features,val_labels)
    plt.show()



if __name__ == '__main__':
    main()