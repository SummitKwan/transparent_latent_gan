""" functions to regress y (labels) based on z (latent space) """

import os
import glob
import numpy as np
import pickle
import h5py
import pandas as pd


import src.misc as misc
import src.tl_gan.feature_axis as feature_axis

##
""" get y and z from pre-generated files """
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
""" regression: use latent space z to predict features y """
feature_slope = feature_axis.find_feature_axis(z, y, method='tanh')

##
""" normalize the feature vectors """
yn_normalize_feature_direction = True
if yn_normalize_feature_direction:
    feature_direction = feature_axis.normalize_feature_axis(feature_slope)
else:
    feature_direction = feature_slope

""" save_regression result to hard disk """
if not os.path.exists(path_feature_direction):
    os.mkdir(path_feature_direction)

pathfile_feature_direction = os.path.join(path_feature_direction, 'feature_direction_{}.pkl'.format(misc.gen_time_str()))
dict_to_save = {'direction': feature_direction, 'name': y_name}
with open(pathfile_feature_direction, 'wb') as f:
    pickle.dump(dict_to_save, f)


##
""" disentangle correlated feature axis """
pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]

with open(pathfile_feature_direction, 'rb') as f:
    feature_direction_name = pickle.load(f)

feature_direction = feature_direction_name['direction']
feature_name = np.array(feature_direction_name['name'])

len_z, len_y = feature_direction.shape


feature_direction_disentangled = feature_axis.disentangle_feature_axis_by_idx(
    feature_direction, idx_base=range(len_y//4), idx_target=None)

feature_axis.plot_feature_cos_sim(feature_direction_disentangled, feature_name=feature_name)

##

