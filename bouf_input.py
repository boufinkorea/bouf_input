import cv2
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt


DS_Path = '/home/jinhkim/workspace/CubeDataset/dataset_scaled/'
Img_FileName = 'img/patchImg%04d.bmp' % (0)
Depth_FileName = 'depth_txt/patchDepth%04d.txt' % (0)
print(DS_Path+Img_FileName)

imgMat = cv2.imread(DS_Path+Img_FileName)

depth_txt = open(DS_Path+Depth_FileName, 'r')
s = depth_txt.readline()
l = (s.split())
row = int(l[0])
col = int(l[1])
print(row, col)
depthMat = numpy.zeros((int(row),int(col)),dtype=numpy.float)

for i in range(0,row):
    s = depth_txt.readline()
    l = s.split()
    for j in range(0,col):
        depthMat[i,j] = float(l[j])
depth_txt.close();


inPatch = numpy.zeros((row,col,4),dtype=numpy.float)

for i in range(0,row):
    for j in range(0,col):
        inPatch[i, j, 0] = float(imgMat[i, j, 0])
        inPatch[i, j, 1] = float(imgMat[i, j, 1])
        inPatch[i, j, 2] = float(imgMat[i, j, 2])
        inPatch[i, j, 3] = depthMat[i,j]

print(inPatch.size)
