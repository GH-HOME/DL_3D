import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.misc import imresize
import cv2
from matplotlib import pyplot as plt
import os
import sys
import demon_run as demon
#
# img1 = Image.open('0000.png')
# img2 = Image.open('0019.png')
# output_depth=demon.run_two_image(img1,img2)
# plt.imshow(output_depth, cmap='Greys')
# plt.show()
# cv2.imwrite('test.png',output_depth)



count = -1;
dir_path='/home/dx/Project/caffe/fusionseg/appearance/images/ILSVRC2015/Data/VID/train'
for dirpath in os.walk(dir_path):
    count=count+1;
    if count==0:
        continue

    str_dir=dirpath[0]+'/'
    files=os.listdir(str_dir)
    for index in range(len(files)):
        if not os.path.isdir(files[index]) and index!=len(files)-1:
            path1=str_dir+files[index]
            path2=str_dir+files[index+1]
            print(path1)
            img1 = Image.open(path1)
            img2 = Image.open(path2)
            output_depth=demon.run_two_image(img1,img2)
            plt.imshow(output_depth, cmap='Greys')
            plt.show()
