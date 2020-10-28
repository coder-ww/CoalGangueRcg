import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import mylib
import PreProcessing as pp
import FeatExtraction as fe
import SupVecMech as mysvm


# 简单画个图
def Visualize(num):  # 可以尝试在柱状图上带数据
    x = range(6)
    plt.xlim(-1, 6)
    plt.ylim(0, 1)
    plt.xticks(range(6), np.linspace(0, 6, 6, dtype=int))
    plt.ylabel("Precision")
    plt.xlabel("Group No.")
    plt.title("Precision of Cross-Validation")
    plt.bar(x, num)
    plt.show()


# 生成训练、测试数据集
def GenSet(times, iteration):
    print("分组:" + str(times))

    coal_prefix = 'D:\\418_2\\coal\\'
    gangue_prefix = 'D:\\418_2\\gangue\\'
    coal_num = 199
    gangue_num = 199

    suffix = '.jpg'
    train = []
    label = []
    testset = []
    for i in range(coal_num):
        path = coal_prefix + str(i) + suffix
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = pp.prep(img)
        # 这里使用了区域关系重采样做插值
        img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
        if i % iteration != times:
            train.append(fe.Rotation_invariant_LBP(img))
            label.append("coal")
        else:
            t0 = time.time()
            tmp = [i, fe.Rotation_invariant_LBP(img), "coal"]
            t1 = time.time()
            tmp.append(t1 - t0)
            testset.append(tmp)
    for i in range(gangue_num):
        path = gangue_prefix + str(i) + suffix
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = pp.prep(img)
        # 这里使用了区域关系重采样做插值
        img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
        if i % iteration != times:
            train.append(fe.Rotation_invariant_LBP(img))
            label.append("gangue")
        else:
            t0 = time.time()
            tmp = [i, fe.Rotation_invariant_LBP(img), "gangue"]
            t1 = time.time()
            tmp.append(t1 - t0)
            testset.append(tmp)
    print(str(times) + "分组完成！")
    return np.asarray(train), np.asarray(label), testset


# 计算每一次交叉检验的准确率
def CalcPre(clf, test):
    cnt = 0
    for i in range(len(test)):
        tmp = [test[i][0], test[i][2], test[i][3]]
        flag = False
        t0 = time.time()
        if clf.predict([test[i][1]]) == test[i][2]:
            cnt += 1
            flag = True
        t1 = time.time()
        tmp[2] += t1 - t0
        tmp.append(flag)
        mylib.record('418_lbp', tmp)
    print("准确率计算完成")
    return cnt / len(test)


def CrossVld():
    precision = []
    iteration = 6
    for i in range(iteration):
        train, label, testset = GenSet(i, iteration)
        clf = mysvm.Train(train, label)
        precision.append(CalcPre(clf, testset))
    return precision