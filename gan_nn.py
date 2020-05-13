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
import gan_utils as utils

from IPython import display

def make_generator_model(input_neurons_count):
    net_config = gan_io.readConfigFile('./configuration/network-model.conf')

    model = tf.keras.Sequential()
    model.add(layers.Dense(utils.multiplyList(net_config['GENERATOR_INPUT_DIMENSIONS']), use_bias=False, input_shape=(utils.multiplyList(net_config['INPUT_IMAGE_DIMENSIONS']),)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape(net_config['GENERATOR_INPUT_DIMENSIONS']))
    assert model.output_shape == (None, 30, 30, 20) # Note: None is the batch size

    i=0
    for neuron_count in net_config['GENERATOR_CONV_NEURONS_COUNT']:
        stride = 1
        if i < 2:
            stride = 2
            i = i + 1
        model.add(layers.Conv2DTranspose(neuron_count, (5, 5), strides=(stride, stride), padding='same', use_bias=False))
        # assert model.output_shape == (None, 30, 30, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 120, 120, 1)

    return model

def make_discriminator_model(neuron_shape):
    net_config = gan_io.readConfigFile('./configuration/network-model.conf')

    hidden_neurons = net_config['DISCRIMINATOR_CONV_NEURONS_COUNT']

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(hidden_neurons[0], (5, 5), strides=(2, 2), padding='same', input_shape=net_config['DISCRIMINATOR_INPUT_DIMENSIONS']))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    i = 0
    for neurons_count in hidden_neurons:
        if i > 0:
            model.add(layers.Conv2D(neurons_count, (5, 5), strides=(2, 2), padding='same'))
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.3))
        i = i + 1

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)