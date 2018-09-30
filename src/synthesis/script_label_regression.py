""" functions to regress y (labels) based on z (latent space) """

import os
import glob
import sys
import numpy as np
import time
import pickle
import datetime
import h5py
import pandas as pd
import tensorflow as tf
import PIL

def gen_time_str():
    """ tool function """
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


##
""" get y and z """
path_gan_sample_img = './asset_results/pggan_celeba_sample_jpg/'
path_celeba_att = './data/raw/celebA_annotation/list_attr_celeba.txt'
path_feature_direction = './asset_results/pg_gan_celeba_feature_direction_40'

filename_sample_y = 'sample_y.h5'
filename_sample_z = 'sample_z.h5'

pathfile_y = os.path.join(path_gan_sample_img, filename_sample_y)
pathfile_z = os.path.join(path_gan_sample_img, filename_sample_z)

with h5py.File(pathfile_y, 'r') as f:
    y = f['y'][:]
with h5py.File(pathfile_z, 'r') as f:
    z = f['z'][:]

# read feature name
df_attr = pd.read_csv(path_celeba_att, sep='\s+', header=1, index_col=0)
y_name = df_attr.columns.values.tolist()

##
""" regression """
reg_res = np.linalg.lstsq(z, y)
feature_slope = reg_res[0]
feature_direction = feature_slope / np.std(feature_slope, axis=0, keepdims=True)

# save_regression result
if not os.path.exists(path_feature_direction):
    os.mkdir(path_feature_direction)

# save to hard disk
pathfile_feature_direction = os.path.join(path_feature_direction, 'feature_direction_{}.pkl'.format(gen_time_str()))
dict_to_save = {'direction': feature_direction, 'name': y_name}
with open(pathfile_feature_direction, 'wb') as f:
    pickle.dump(dict_to_save, f)


