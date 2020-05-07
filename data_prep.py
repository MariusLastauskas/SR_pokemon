from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import gan_io
import gan_image_transformations as imtr
# import gan_nn as nn
import imageio

from IPython import display

train_images_path = './train_images/'
train_images_file = './prepaired_data/train_images.dat'
train_mini_images_file = './prepaired_data/train_mini_images.dat'

train_list = []
# kk = 0

for f in glob.glob("./train_images/*.png"):
    # if kk == 10:
        # break
    # im = imageio.imread(f)
    im = PIL.Image.open(f).convert('L')
    im = np.array(im)
    # plt.imshow(im)
    # plt.show()
    train_list.append(im)
    # kk = kk + 1
    # print(im)
    # plt.imshow(im)
    # plt.show()

gan_io.save_list(train_list, './prepaired_data/train.dat', 255)
# list = gan_io.load_list('./prep_data/train.dat')

mini_list = []

for image in train_list:
    mini_image = imtr.image_minify(image)
    mini_list.append(mini_image)

gan_io.save_list(mini_list, './prepaired_data/train_mini.dat', 255)

# print(image2[60][60])

# print(list[1][60][60])
# print(np.array(image2).shape)
# print(np.array(list[1]).shape) 
# plt.imshow(list[1])
# plt.show()

 
# plt.imshow(imtr.image_minify(image2))
# plt.show()