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
import gan_nn as nn

from IPython import display

def generate(image_path):
    image = PIL.Image.open(image_path).convert('L')
    image = np.array(image)
    
    im = []
    for row in image:
        r = []
        for pixel in row:
            r.append(pixel / (255 / 2) - 1)
        im.append(r)


    generator = nn.make_generator_model(60 * 60)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    generated_image = generator(tf.reshape(im, [1, 60 * 60]), training=False)
    
    plt.imshow(tf.reshape(generated_image, [120, 120]), cmap='gray')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('./uploads/enhanced.png', bbox_inches = 'tight', pad_inches = 0)
    return 'enhanced.png'
