import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def IfOverExposed(img):
    hist, bins = np.histogram(img, 256, range=(0, 256))
    print(hist)
    bins = np.delete(bins, 256)
    print(bins)

    plt.bar(bins, hist)
    plt.show()
    # return True
