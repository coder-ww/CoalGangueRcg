# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import SupVecMech as mysvm
import PreProcessing as pp
import FeatExtraction as fe
import CrsVldt as cvd
import mylib
import cv2
import numpy as np
import pickle

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    precision = []
    precision = cvd.CrossVld()
    for i in range(len(precision)):
        print(precision[i])
    cvd.Visualize(precision)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
