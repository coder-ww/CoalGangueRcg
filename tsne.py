from time import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

import FeatExtraction as fe


def get_data():
    coal = [0,10,20,40,50,60,70,80,90,100,241,251,261,271,281,291,301,311,321,331,102,112,122,232,242,252,262,272,282,342,352,362,372,382,143,153,163,173,183,193]
    gangue = [0,10,20,30,40,50,60,70,80,210,161,171,181,191,201,211,221,231,241,251,3,13,23,33,43,53,4,14,24,184,194,204,214,224,234,244,254,264,274,284]
    fet = []
    lab = []
    print("coal")
    for i in range(40):
        print(i)
        path = 'D:\\20201103\\20191218-01\\pic\\coal\\'+str(coal[i])+".jpg"
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(500,500),interpolation=cv2.INTER_AREA)
        vec = fe.Hog(img)
        print(len(vec))
        fet.append(vec)
        lab.append("coal")
    print("gangue")
    for i in range(40):
        print(i)
        path = 'D:\\20201103\\20191218-01\\pic\\gangue\\'+str(gangue[i])+".jpg"
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (500,500), interpolation=cv2.INTER_AREA)
        vec = fe.Hog(img)
        print("gangue: %d " % len(vec))
        fet.append(vec)
        lab.append("gangue")
    data = np.array(fet,dtype=float)
    label = np.array(lab, dtype=str)
    return data, label


def plot_embedding(data, label, title):
    fig = plt.figure()
    coal_x = []
    coal_y = []
    gangue_x = []
    gangue_y = []
    for i in range(data.shape[0]):
        if label[i] == "coal":
            coal_x.append(data[i][0])
            coal_y.append(data[i][1])
        else:
            gangue_x.append(data[i][0])
            gangue_y.append(data[i][1])
    plt.scatter(coal_x,coal_y,marker='o',label = "coal")
    plt.scatter(gangue_x,gangue_y,marker='*',label="gangue")
    plt.legend()
    return fig


def main():
    data, label = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the CoalGangue (time %.2fs)'
                         % (time() - t0))
    plt.show()


if __name__ == '__main__':
    main()
