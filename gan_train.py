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

def fit(epochs_count=1000):
    print('###### Load train data')
    train_data = gan_io.load_list('./prepaired_data/train.dat')
    print('###### Load minified train data')
    train_mini = gan_io.load_list('./prepaired_data/train_mini.dat')

    BUFFER_SIZE = 60000
    BATCH_SIZE = 25

    # Batch and shuffle the data
    print('###### Batch and shuffle the data')
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_mini)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    ######## Main ########
    # inst models
    print('###### Instantiating models')

    generator = nn.make_generator_model(60 * 60)
    discriminator = nn.make_discriminator_model([180, 120, 1])

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004, beta_1=0.5)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    EPOCHS = epochs_count
    num_examples_to_generate = 16

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    preview_input = train_mini[0:num_examples_to_generate]

    # create a line plot of loss for the gan and save to file
    def plot_history(gen_loss, disc_loss):
        plt.plot(gen_loss, label='gen_loss')
        plt.plot(disc_loss, label='disc_loss')
        plt.legend()
        # save plot to file
        plt.savefig('./results_baseline/plot_line_plot_loss.png')
        plt.close()

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(tf.reshape(images[1], [len(images[1]), 60 * 60]), training=True)
            generated_images = tf.reshape(generated_images, [len(generated_images), 120, 120, 1])

            real_image_tensors = []
            fake_image_tensors = []

            for i in range(images[0].get_shape()[0]):
                mini_with_padding = tf.pad(images[1][i][0], [[0, 0], [30, 30]], 'CONSTANT')
                concationated_real_image = tf.concat([images[0][i][0], mini_with_padding], 0)
                concationated_fake_image = tf.concat([tf.reshape(generated_images[i], [120, 120]), mini_with_padding], 0)
                reshaped_real_image = tf.reshape(concationated_real_image, [180, 120, 1])
                reshaped_fake_image = tf.reshape(concationated_fake_image, [180, 120, 1])
                real_image_tensors.append(reshaped_real_image)
                fake_image_tensors.append(reshaped_fake_image)
                
            real_output_images = tf.stack(real_image_tensors)
            fake_output_images = tf.stack(fake_image_tensors)

            real_output = discriminator(real_output_images, training=True)
            fake_output = discriminator(fake_output_images, training=True)

            gen_loss = nn.generator_loss(fake_output)
            disc_loss = nn.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return (gen_loss, disc_loss)

    def train(dataset, epochs):
        gen_loss_hist, disc_loss_hist = list(), list()
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                gen_loss, disc_loss = train_step(image_batch)
                gen_loss_hist.append(gen_loss)
                disc_loss_hist.append(disc_loss)

            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            gan_io.generate_and_save_images(generator,
                                    epoch + 1,
                                    tf.reshape(preview_input, [num_examples_to_generate, 60 * 60]))
            if (epoch + 1) % 2 == 0:
                plot_history(gen_loss_hist, disc_loss_hist)
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        gan_io.generate_and_save_images(generator,
                                epochs,
                                tf.reshape(preview_input, [num_examples_to_generate, 60 * 60]))

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    train(train_dataset, EPOCHS)