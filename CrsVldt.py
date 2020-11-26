import numpy as np
import gc
import time
import mylib
import FeatExtraction as fe
import SupVecMech as mysvm

# 计算每一组的四个结果
def CalcIndicator(clf, featset, testset):
    tp = fp = tn = fn = 0
    for tmp in testset:
        flag = False
        t0 = time.time()
        if tmp[2] == "coal":
            if clf.predict([featset[tmp[0]]]) == tmp[2]:
                tp += 1
                flag = True
            else:
                tn += 1
        else:
            if clf.predict([featset[tmp[0]]]) == tmp[2]:
                fp += 1
                flag = True
            else:
                fn += 1
        t1 = time.time()
        tmp[3] += t1 - t0
        tmp.append(flag)
        del tmp[0]
        mylib.Record('neimeng55', tmp)
    return tp, fp, tn, fn

# 交叉检验代码
def CrossVld(iteration, num = 390):
    print("识别石块总个数 %d" % (2*num))
    precision = []
    recall = []
    accuracy = []
    train, label, pos, time = fe.GenFeatSet(num)
    print("特征计算完成")
    for i in range(iteration):
        print("开始第%d组检验" % i)
        tmp_t = []
        tmp_l = []
        testset = []
        for j in range(2*num):
            if j%iteration == i:
                tmp = [j, pos[j], label[j], time[j]]
                testset.append(tmp)
            else:
                tmp_t.append(train[j])
                tmp_l.append(label[j])
        # 这里可以加入读写模型操作
        clf = mysvm.Train(np.asarray(tmp_t), np.asarray(tmp_l))
        tp, fp, tn, fn = CalcIndicator(clf, train, testset)
        print("指标计算完成")
        accuracy.append((tp+tn)/(2*num))
        precision.append(tp/(tp+fp))
        recall.append(tp/(tp+fn))
    del train, tmp_t, tmp_l, label, pos, time, testset
    gc.collect()
    return accuracy, precision, recall
