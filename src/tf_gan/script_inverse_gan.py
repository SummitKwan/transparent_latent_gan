""" script to test out the inverse GAN """

import os
import glob
import sys
import pickle

import numpy as np
import PIL
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt

import src.tf_gan.generate_image as generate_image


path_inverse_gan_sample = './asset_results/inverse_gan_sample/'
if not os.path.exists(path_inverse_gan_sample):
    os.mkdir(path_inverse_gan_sample)


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True

##
""" generate several images """
yn_generate_samples = True
num_sample = 32

np.random.seed(0)
with tf.Session(config=config) as sess:
    with open(generate_image.path_model, 'rb') as file:
        G, D, Gs = pickle.load(file)

    for i in range(num_sample):
        print(i)
        z = np.random.randn(512)
        img = generate_image.gen_single_img(Gs=Gs, z=z)
        generate_image.save_img(img, os.path.join(path_inverse_gan_sample, 'image_{:0>6}.png'.format(i)))
        np.save(os.path.join(path_inverse_gan_sample, 'z_{:0>6}.png'.format(i)), img)

##
config = tf.ConfigProto(allow_soft_placement=True)
# sess = tf.InteractiveSession(config=config)
with tf.Session(config=config) as sess:

    with open(generate_image.path_model, 'rb') as file:
        G, D, Gs = pickle.load(file)

    G.print_layers()
    G.list_layers()
    # tf.get_variable('latens_in')

    temp = tf.train.Saver().save(sess, './asset_model/temp_GS.ckpt')

##
sess.close()
