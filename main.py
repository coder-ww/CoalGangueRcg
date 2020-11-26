import mylib
import CrsVldt as cvd

if __name__ == '__main__':
    # k次k折交叉检验
    times = 10
    acc = []
    pre = []
    rec = []
    for i in range(times):
        accuracy, precision, recall = cvd.CrossVld(times)
        acc.append(sum(accuracy)/len(accuracy))
        pre.append(sum(precision)/len(precision))
        rec.append(sum(recall)/len(recall))
    mylib.Visualize(acc, times)
    mylib.Visualize(pre, times)
    mylib.Visualize(rec, times)