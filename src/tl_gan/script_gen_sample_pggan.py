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


""" gen samples and save as pickle """

n_batch = 8000
batch_size = 32

with tf.Session() as sess:

    # Import official CelebA-HQ networks.
    try:
        with open(path_model, 'rb') as file:
            G, D, Gs = pickle.load(file)
    except FileNotFoundError:
        print('before running the code, download pre-trained model to project_root/asset_model/')
        raise

    # Generate latent vectors.
    # latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
    # latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10

    for i_batch in range(n_batch):
        try:
            i_sample = i_batch * batch_size

            tic = time.time()

            latents = np.random.randn(batch_size, *Gs.input_shapes[0][1:])

            # Generate dummy labels (not used by the official networks).
            labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

            # Run the generator to produce a set of images.
            images = Gs.run(latents, labels)

            images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
            images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

            images = images[:, 4::8, 4::8, :]

            with open(os.path.join(path_gen_sample, 'pggan_celeba_{:0>6d}.pkl'.format(i_sample)), 'wb') as f:
                pickle.dump({'z': latents, 'x': images}, f)

            toc = time.time()
            print(i_sample, toc-tic)

        except:
            print('error in {}'.format(i_sample))


""" view generated samples """
yn_view_sample = False
if yn_view_sample:
    with open(os.path.join(path_gen_sample, 'pggan_celeba_{:0>6d}.pkl'.format(0)), 'rb') as f:
        temp = pickle.load(f)

    import matplotlib.pyplot as plt
    plt.imshow(temp['x'][0]); plt.show()


