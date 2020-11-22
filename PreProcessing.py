import cv2

def Gamma(img):#伽马矫正
    s = img.shape
    for i in range(s[0]):
        for j in range(s[1]):
            img[i][j] = img[i][j]**1.1
    return img

def HistEqual(img):#分区域直方图均衡化，用于图像增强
    #这个参数可以调
    clash = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clash.apply(img)
    return cl

# 调整顺序，寻求最优方案
def Prep(img):
    # 高斯降噪
    img = cv2.GaussianBlur(img,(3,3),0)
    # 拉普拉斯增强
    img = cv2.Laplacian(img,-1,ksize = 3)
    # 分区域直方图均衡化
    img = HistEqual(img)
    # 开运算
    g = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, g)
    return img