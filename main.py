import mylib
import CrsVldt as cvd

if __name__ == '__main__':
    # k次k折交叉检验
    times = 10
    for i in range(times):
        accuary, precision, recall = cvd.CrossVld(times)

    mylib.Visualize(accuary, times, i)
    mylib.Visualize(precision, times, i)
    mylib.Visualize(recall, times, i)