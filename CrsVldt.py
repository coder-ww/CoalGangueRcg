import numpy as np
import gc
import cv2
import time
import mylib
import PreProcessing as pp
import FeatExtraction as fe
import SupVecMech as mysvm

def GenFeatSet():
    coal_prefix = 'D:\\20201103\\20191218-01\\pic\\coal\\'
    gangue_prefix = 'D:\\20201103\\20191218-01\\pic\\gangue\\'
    suffix = '.jpg'

    train = []
    label = []

    num = 390
    cand_c, cand_g = mylib.GenCandidate(num)
    for index in cand_c:
        path = coal_prefix + str(index) + suffix
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # 这里使用了区域关系重采样做插值
        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
        img = pp.Prep(img)
        train.append(fe.Hog(img))
        label.append("coal")
    for index in cand_g:
        path = gangue_prefix + str(index) + suffix
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # 这里使用了区域关系重采样做插值
        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
        img = pp.Prep(img)
        train.append(fe.Hog(img))
        label.append("gangue")
    return train, label

# 生成训练、测试数据集
def GenSet(times, iteration):
    print("分组:" + str(times))

    coal_prefix = 'D:\\20201103\\20191218-01\\pic\\coal\\'
    gangue_prefix = 'D:\\20201103\\20191218-01\\pic\\gangue\\'
    suffix = '.jpg'

    # 期望选取的煤/矸石图像数量
    num = 390
    cand_c, cand_g = mylib.GenCandidate(num)

    train = []
    label = []
    testset = []

    for i in range(num):
        path = coal_prefix + str(cand_c[i]) + suffix
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = pp.Prep(img)
        # 这里使用了区域关系重采样做插值
        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
        if i % iteration != times:
            train.append(fe.Hog(img))
            label.append("coal")
        else:
            t0 = time.time()
            tmp = [i, fe.Hog(img), "coal"]
            t1 = time.time()
            tmp.append(t1 - t0)
            testset.append(tmp)
    for i in range(num):
        path = gangue_prefix + str(cand_g[i]) + suffix
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = pp.Prep(img)
        # 这里使用了区域关系重采样做插值
        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
        if i % iteration != times:
            train.append(fe.Hog(img))
            label.append("gangue")
        else:
            t0 = time.time()
            tmp = [i, fe.Hog(img), "gangue"]
            t1 = time.time()
            tmp.append(t1 - t0)
            testset.append(tmp)
    print(str(times) + "分组完成！")
    return np.asarray(train), np.asarray(label), testset

# 计算每一组的四个结果
def CalcPre(clf, test):
    tp = fp = tn = fn = 0
    for i in range(len(test)):
        tmp = [test[i][0], test[i][2], test[i][3]]
        flag = False
        t0 = time.time()
        if test[i][2] == "coal":
            if clf.predict([test[i][1]]) == test[i][2]:
                tp += 1
                flag = True
            else:
                tn += 1
        else:
            if clf.predict([test[i][1]]) == test[i][2]:
                fp += 1
                flag = True
            else:
                fn += 1
        t1 = time.time()
        tmp[2] += t1 - t0
        tmp.append(flag)
        mylib.Record('neimeng55', tmp)
    print("准确率计算完成")
    return tp,fp,tn,fn

def CrossVld():
    precision = []
    iteration = 10
    for i in range(iteration):
        train, label, testset = GenSet(i, iteration)

        clf = mysvm.Train(train, label)
        precision.append(CalcPre(clf, testset))
    return precision
