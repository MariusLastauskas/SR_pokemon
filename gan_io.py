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

from IPython import display

def save_list(list, file_path, normalization_factor = 1):
    with open(file_path, 'w') as writer:
        for item in list:
            writer.write('[')
            for vector in item:
                writer.write('[')
                for pixel in vector:
                    writer.write('%f ' % (pixel / (normalization_factor / 2) - 1))
                writer.write(']')
            writer.write(']\n')

def load_list(file_path):
    kk = 0
    list = []
    with open(file_path, 'r') as reader:
        for item in reader:
            kk = kk + 1
            if (kk == 2000):
                break
            if ((kk + 1) % 100 == 0):
                print('item', kk)
            image = []
            for v in item.split(']][['):
                vector = []
                for p in v.split(']['):
                    pixel = []
                    for rgba in p.replace('[', '').replace(']', '').split(' '):
                        if len(rgba) > 0 and rgba != "\n":                        
                            pixel.append(float(rgba))
                    vector.append(pixel)
                image.append(vector)
            list.append(image)
            # print(np.array(image).shape)
    return list

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    print('################### generate and save image epch: ', epoch)
    
    predictions = model(test_input, training=False)
    print('################### generated image')
    predictions = tf.reshape(predictions, [len(predictions), 120, 120])

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :] + 1) * 255 / 2, cmap='gray')
        plt.axis('off')

    plt.savefig('./results/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

def readConfigFile(file):
    config = {}
    with open(file, 'r') as reader:
        for item in reader:
            parts = item.split(':')
            if len(parts) == 2:
                value = parts[1].strip().split(' ')
                for i in range(len(value)):
                    try:
                        value[i] = int(value[i])
                    except:
                        value[i] = value[i]
                config[parts[0]] = value
    return config
