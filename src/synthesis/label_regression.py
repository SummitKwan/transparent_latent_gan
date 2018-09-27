""" functions to regress y (labels) based on z (latent space) """

import os
import glob
import sys
import numpy as np
import scipy as sp
import pickle
import datetime
import h5py
import pandas as pd

##
""" get y and z """
path_gan_sample_img = './asset_results/pggan_celeba_sample_jpg/'

filename_sample_y = 'sample_y.h5'
filename_sample_z = 'sample_z.h5'

pathfile_y = os.path.join(path_gan_sample_img, filename_sample_y)
pathfile_z = os.path.join(path_gan_sample_img, filename_sample_z)

with h5py.File(pathfile_y, 'r') as f:
    y = f['y'][:]
with h5py.File(pathfile_z, 'r') as f:
    z = f['z'][:]

path_celeba_att = './data/raw/celebA_annotation/list_attr_celeba.txt'
df_attr = pd.read_csv(path_celeba_att, sep='\s+', header=1, index_col=0)
y_name = df_attr.columns.values.tolist()

##
""" regression """
reg_res = np.linalg.lstsq(z, y)
feature_slope = reg_res[0]
feature_direction = feature_slope / np.std(feature_slope, keepdims=True)

##
""" visualized how it affects image synthesis """

import matplotlib.pyplot as plt


##
""" test_discovered features """

# path to model code and weight
path_pg_gan_code = './src/model/pggan'
path_model = './asset_model/karras2018iclr-celebahq-1024x1024.pkl'
sys.path.append(path_pg_gan_code)

path_gan_explore = './asset_results/pggan_celeba_feature_axis_explore/'
if not os.path.exists(path_gan_explore):
    os.mkdir(path_gan_explore)

""" play with the latent space """
sess = tf.InteractiveSession()

try:
    with open(path_model, 'rb') as file:
        G, D, Gs = pickle.load(file)
except FileNotFoundError:
    print('before running the code, download pre-trained model to project_root/asset_model/')
    raise

batch_size = 3

##
latents_c = np.random.randn(1, *Gs.input_shapes[0][1:])

for i_feature in range(feature_direction.shape[1]):
    latents_0 = latents_c - feature_direction[:, i_feature][None, :]*0.02
    latents_1 = latents_c + feature_direction[:, i_feature][None, :]*0.02

    print(np.mean(latents_0-latents_1)**2)

    latents = np.random.randn(batch_size, *Gs.input_shapes[0][1:])
    for i_alpha, alpha in enumerate(np.linspace(0, 1, batch_size)):
        latents[i_alpha, :] = latents_0[0]*(1-alpha) + latents_1[0]*alpha

    # Generate dummy labels (not used by the official networks).
    labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

    # Run the generator to produce a set of images.
    images = Gs.run(latents, labels)

    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

    images = images[:, 2::4, 2::4]

    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save images as PNG.
    for idx in range(images.shape[0]):
        PIL.Image.fromarray(images[idx], 'RGB')\
            .save(os.path.join(path_gan_explore,
                               'img_{}_{}_{}_{}.png'.format(time_str, i_feature, y_name[i_feature], idx)))
    np.save(os.path.join(path_gan_explore, 'img_{}_{}.pkl'.format(time_str, i_feature)), labels)
