""" web gui using bokeh, not fulling functional yet """


import os
import glob
import sys
import numpy as np
import time
import pickle
import tensorflow as tf
import random

import bokeh
from bokeh.layouts import column
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc

sys.path.append('.')
import src.tl_gan.feature_axis as feature_axis

""" load feature directions """
path_feature_direction = './asset_results/pg_gan_celeba_feature_direction_40'

pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]

with open(pathfile_feature_direction, 'rb') as f:
    feature_direction_name = pickle.load(f)

feature_direction = feature_direction_name['direction']
feature_name = feature_direction_name['name']
num_feature = feature_direction.shape[1]

""" load gan model """

# path to model code and weight
path_pg_gan_code = './src/model/pggan'
path_model = './asset_model/karras2018iclr-celebahq-1024x1024.pkl'
sys.path.append(path_pg_gan_code)


""" create tf session """
yn_CPU_only = False

if yn_CPU_only:
    config = tf.ConfigProto(device_count = {'GPU': 0}, allow_soft_placement=True)
else:
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)

try:
    with open(path_model, 'rb') as file:
        G, D, Gs = pickle.load(file)
except FileNotFoundError:
    print('before running the code, download pre-trained model to project_root/asset_model/')
    raise

num_latent = Gs.input_shapes[0][1]

latents = np.random.randn(1, *Gs.input_shapes[0][1:])
# Generate dummy labels
dummies = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])


def gen_image(latents):
    """
    tool funciton to generate image from latent variables
    :param latents: latent variables
    :return:
    """
    images = Gs.run(latents, dummies)
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
    return images[0]

img_cur = gen_image(latents)


##
# create a plot and style its properties
def get_img_for_bokeh(img):
    H, W, _ = img.shape
    img_bokeh = np.empty([H, W], dtype=np.uint32)
    img_bokeh_view = img_bokeh.view(dtype=np.uint8).reshape((H, W, 4))
    img_bokeh_view[:, :, :3] = np.flipud(img)
    img_bokeh_view[:, :, 3] = 255
    return img_bokeh


p = figure(x_range=(0, 128), y_range=(0, 128), toolbar_location=None)

p.image_rgba(image=[get_img_for_bokeh(img_cur)], x=0, y=0, dw=128, dh=128)

curdoc().add_root(p)

# bokeh.plotting.show(p)