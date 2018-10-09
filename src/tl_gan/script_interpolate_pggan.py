"""
try face tl_gan using pg-gan model, modified from
https://drive.google.com/drive/folders/1A79qKDTFp6pExe4gTSgBsEOkxwa2oes_
"""

"""
prerequsit: before running the code, download pre-trained model to project_root/asset_model/
pretrained model download url: https://drive.google.com/drive/folders/15hvzxt_XxuokSmj0uO4xxMTMWVc0cIMU
model name: karras2018iclr-celebahq-1024x1024.pkl
"""

import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import datetime

# path to model code and weight
path_pg_gan_code = './src/model/pggan'
path_model = './asset_model/karras2018iclr-celebahq-1024x1024.pkl'
sys.path.append(path_pg_gan_code)

# path to model generated results
path_gen_sample = './asset_results/pggan_celeba_sample_pkl/'
if not os.path.exists(path_gen_sample):
    os.mkdir(path_gen_sample)
path_gan_explore = './asset_results/pggan_celeba_explore/'
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

batch_size = 8

latent_mode = 'random'
if latent_mode == 'random':
    latents_0 = np.random.randn(1, *Gs.input_shapes[0][1:])
    latents_1 = np.random.randn(1, *Gs.input_shapes[0][1:])
elif latent_mode == 'scale':
    latents_0 = np.random.randn(1, *Gs.input_shapes[0][1:]) * 3
    latents_1 = latents_0*(-1)
else:
    raise Exception('latent mode not accepted')

print(np.sum(latents_0-latents_1)**2)

latents = np.random.randn(batch_size, *Gs.input_shapes[0][1:])
for i_alpha, alpha in enumerate(np.linspace(0, 1, batch_size)):
    latents[i_alpha, :] = latents_0[0]*alpha + latents_1[0]*(1-alpha)

# Generate dummy labels (not used by the official networks).
labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

# Run the generator to produce a set of images.
images = Gs.run(latents, labels)

images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save(os.path.join(path_gan_explore,
                                                              'img_{}_{}_{}.png'.format(latent_mode, time_str, idx)))
np.save(os.path.join(path_gan_explore, 'img_{}_{}.pkl'.format(latent_mode, time_str)), labels)


sess.close()
