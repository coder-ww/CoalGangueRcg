from xml.dom.minidom import parse
import xml.dom.minidom
import cv2

def GenDataset():
    cnt_s = 0
    cnt_c = 0
    dir = "D:\\BaiduNetdiskDownload\\raw\\"
    for i in range(328):
        path_xml = dir + str(i) + '.xml'
        path_img = dir + str(i) + '.jpg'
        img = cv2.imread(path_img)
        DOMTree = xml.dom.minidom.parse(path_xml)
        annotation = DOMTree.documentElement
        objects = annotation.getElementsByTagName("object")
        for ob in objects:
            name = ob.getElementsByTagName("name")[0]
            print("name: %s" % name.childNodes[0].data)
            bndbox = ob.getElementsByTagName("bndbox")[0]
            xmin = bndbox.getElementsByTagName("xmin")[0]
            minx = int(xmin.childNodes[0].data)
            xmax = bndbox.getElementsByTagName("xmax")[0]
            maxx = int(xmax.childNodes[0].data)
            ymin = bndbox.getElementsByTagName("ymin")[0]
            miny = int(ymin.childNodes[0].data)
            ymax = bndbox.getElementsByTagName("ymax")[0]
            maxy = int(ymax.childNodes[0].data)
            img_clip = img[miny:maxy, minx:maxx]
            if name.childNodes[0].data == "coal":
                cv2.imwrite("D:\\418\\coal\\"+str(cnt_c)+".jpg",img_clip)
                cnt_c += 1
            else:
                cv2.imwrite("D:\\418\\gangue\\"+str(cnt_s)+".jpg",img_clip)
                cnt_s += 1

GenDataset()
