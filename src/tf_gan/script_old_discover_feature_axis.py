""" script to discover feature axis in the latent space """

"""
pre-requisite: this code needs pre-generated feature-image pairs, stored as pickle files located at:
project_root/asset_results/pggan_celeba_sample_pkl
"""


import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import datetime
import glob


# path to model generated results
path_gan_sample = './asset_results/pggan_celeba_sample_pkl/'
if not os.path.exists(path_gan_sample):
    os.mkdir(path_gan_sample)

##
""" function to get features """
def get_feature(x):
    """
    get a list of features from images

    :param x: generated images, of shape [num_images, height, width, rgb]
    :return: feature table, of shape [num_images, num_features]
    """

    n, h, w, _ = x.shape
    fg_lum = np.mean(x[:, h//4:h//4*3, w//4:w//4*3], axis=(1,2,3))
    bg_lum = np.mean(x[:, :h//4, :w//4], axis=(1,2,3))
    return np.hstack((fg_lum[:, None], bg_lum[:, None]))

##
""" get the simplest feature: dark-bright skin color """
list_pkl = sorted(glob.glob(path_gan_sample+'*.pkl'))

list_z = []
list_y = []

for file_pkl in list_pkl[:8000]:
    with open(file_pkl, 'rb') as f:
        dict_zx = pickle.load(f)
        z = dict_zx['z']
        x = dict_zx['x']
        y = get_feature(x)

        list_z.append(z)
        list_y.append(y)

z_all = np.concatenate(list_z, axis=0)
y_all_raw = np.concatenate(list_y, axis=0)
y_all = (y_all_raw - np.mean(y_all_raw, axis=0, keepdims=True))/np.std(y_all_raw, axis=0, keepdims=True)

##
"""discover feature axis"""
reg_res = np.linalg.lstsq(z_all, y_all)
feature_directon = reg_res[0]


##
""" visualize stored samples """
import matplotlib.pyplot as plt
plt.imshow(x[5]); plt.show()

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

for i_feature in range(feature_directon.shape[1]):
    latents_0 = latents_c - feature_directon[:, i_feature][None, :]*2
    latents_1 = latents_c + feature_directon[:, i_feature][None, :]*2

    print(np.sum(latents_0-latents_1)**2)

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
        PIL.Image.fromarray(images[idx], 'RGB').save(os.path.join(path_gan_explore,
                                                                  'img_{}_{}_{}.png'.format(time_str, i_feature, idx)))
    np.save(os.path.join(path_gan_explore, 'img_{}_{}.pkl'.format(time_str, i_feature)), labels)

##
sess.close()


