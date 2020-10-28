import cv2
import numpy as np
import FeatExtraction as fe
import SupVecMech as mysvm

coal_prefix = 'D:\\coaldata\\coal\\'
gangue_prefix = 'D:\\coaldata\\gangue\\'
coal_num = 408
gangue_num = 396
train = []
label = []
test = []
print("开始读煤了")
for i in range(coal_num):
    path = coal_prefix + str(i) + '.jpg'
    img = cv2.imread(path)
    train.append(fe.Hog(img))
    label.append("coal")
print("开始读矸石了")
for i in range(gangue_num):
    path = gangue_prefix + str(i) + '.jpg'
    img = cv2.imread(path)
    train.append(fe.Hog(img))
    label.append("gangue")
print("模型训练！")
tr = np.asarray(train)
la = np.asarray(label)
clf = mysvm.Train(tr,la)
#保存该模型
print("训练完成啦！")

coaldir = "D:\\418_01\\new\\coal\\"
stonedir = "D:\\418_01\\new\\stone\\"
coalnum = 96
stonenum = 158
cnt = 0
for i in range(coalnum):
    path = coaldir + str(i) +'.jpg'
    img = cv2.imread(path)
    if clf.predict([fe.Hog(img)]) == "coal":
        cnt += 1
print("测试完成一半了")
for i in range(stonenum):
    path = stonedir + str(i) + '.jpg'
    img = cv2.imread(path)
    if clf.predict([fe.Hog(img)]) == "gangue":
        cnt += 1
print("这就是准确率：")
print(cnt/(96+158))