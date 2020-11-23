import os
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook

# 结果可视化
def Visualize(num, groups, index):  # 可以尝试在柱状图上带数据
    x = range(groups)
    plt.xlim(-1, groups)
    plt.ylim(0.5, 1)
    plt.xticks(range(10), np.linspace(0, groups, groups, dtype=int))
    plt.ylabel(index)
    plt.xlabel("Group No.")
    plt.title("%s of Cross-Validation" % index)
    plt.bar(x, num)
    plt.show()

# 重命名文件夹中的文件
def rename():
    path = r'D:\20201103\20191218-01\pic\coal'
    cnt = 0
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    for files in filelist:  # 遍历所有文件
        Olddir = os.path.join(path, files)  # 原来的文件路径
        if os.path.isdir(Olddir):  # 如果是文件夹则跳过
            continue
        filename = os.path.splitext(files)[0]  # 文件名
        filetype = os.path.splitext(files)[1]  # 文件扩展名
        Newdir = os.path.join(path, str(cnt) + filetype)
        os.rename(Olddir, Newdir)
        # if filetype == ".xml":
        cnt += 1

# 将信息记录到excel表中
def record(sheetname, ore):
    wb = load_workbook('dataset.xlsx')
    ws = wb[sheetname]
    ws.append(ore)
    wb.save('dataset.xlsx')

# 在excel表中记录未被选择的图片
def filter():
    path = r'D:\coal-gangue\selected\gangue'
    filelist = os.listdir(path)
    for i in range(457):
        tmp = str(i) + ".jpg"
        if tmp not in filelist:
            record("selected",(i,"coal"))

# filter()
# rename()