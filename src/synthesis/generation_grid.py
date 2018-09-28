""" generation of faces from one center image, and move along every feature axis """

import os
import glob
import sys
import numpy as np
import time
import pickle
import datetime
import tensorflow as tf
import PIL


## load feature directions
path_feature_direction = './asset_results/pg_gan_celeba_feature_direction_40'

pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]

with open(pathfile_feature_direction, 'rb') as f:
    feature_direction_name = pickle.load(f)

feature_direction = feature_direction_name['direction']
feature_name = feature_direction_name['name']

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

batch_size = 7

##
latents_c = np.random.randn(1, *Gs.input_shapes[0][1:])

for i_feature in range(feature_direction.shape[1]):
    latents_0 = latents_c - feature_direction[:, i_feature][None, :]*0.07
    latents_1 = latents_c + feature_direction[:, i_feature][None, :]*0.07

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

    # downsize images
    # images = images[:, 2::4, 2::4]

    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save images as PNG.
    for idx in range(images.shape[0]):
        PIL.Image.fromarray(images[idx], 'RGB')\
            .save(os.path.join(path_gan_explore,
                               'img_{}_{}_{}_{}.png'.format(time_str, i_feature, feature_name[i_feature], idx)))
    np.save(os.path.join(path_gan_explore, 'img_{}_{}.pkl'.format(time_str, i_feature)), labels)

