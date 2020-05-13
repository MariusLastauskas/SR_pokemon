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
import argparse

import gan_io
import gan_nn as nn
import gan_train
import gan_upscale as gen

from IPython import display

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--train', nargs=1, metavar='N-EPOHCS', type=int, help='Run network training process for N-EPOCHS.')
parser.add_argument('--upscale', nargs=1, metavar='IMAGE-PATH', help='Upscale the image in IMAGE-PATH, using pretrained network.')

args = parser.parse_args()

if args.train != None:
  gan_train.fit(args.train[0])

elif args.upscale != None:
  gen.generate(args.upscale[0])