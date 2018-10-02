""" functions to regress y (labels) based on z (latent space) """

import os
import glob
import numpy as np
import pickle
import h5py
import pandas as pd


import src.misc as misc
import src.tf_gan.feature_axis as feature_axis

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
import importlib
importlib.reload(feature_axis)

feature_slope = feature_axis.find_feature_axis(z, y, method='tanh')

##

# feature_direction = feature_axis.normalize_feature_axis(feature_slope)
feature_direction = feature_slope

# save_regression result
if not os.path.exists(path_feature_direction):
    os.mkdir(path_feature_direction)

# save to hard disk
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

##
""" plot correlation between feature axis """

yn_plot_feature_correlation  = False

if yn_plot_feature_correlation == True:

    import matplotlib.pyplot as plt

    def plot_feature_correlation(feature_direction, feature_name):
        feature_correlation = np.corrcoef(feature_direction.transpose())
        len_z, len_y = feature_direction.shape
        c_lim_abs = np.max(np.abs(feature_direction))
        plt.figure(figsize=(8,8))
        plt.pcolormesh(np.arange(len_y+1), np.arange(len_y+1), feature_correlation,
                       cmap='coolwarm', clim=[-c_lim_abs, +c_lim_abs])
        plt.gca().invert_yaxis()
        plt.colorbar()
        # plt.axis('square')
        plt.xticks(np.arange(len_y) + 0.5, feature_name, fontsize='x-small', rotation='vertical')
        plt.yticks(np.arange(len_y) + 0.5, feature_name, fontsize='x-small')
        plt.show()


    plot_feature_correlation(feature_direction, feature_name)

##

