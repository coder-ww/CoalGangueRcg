import numpy as np
from sklearn import svm

np.set_printoptions(threshold=np.inf)

def Train(train,label):
    clf = svm.SVC(C = 1.1)#最好不要使用构造函数的默认参数https://www.cnblogs.com/crawer-1/p/8870700.html
    clf.fit(train,label)
    #保存模型
    #with open(r'C:\Users\wei\source\repos\coal-gangue\clf_default.pickle','wb') as f_o:
    #    pickle.dump(clf,f)
    print("模型训练完成")
    return clf
