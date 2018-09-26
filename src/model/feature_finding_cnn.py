""" train a simple cnn model for abstracting features from celebA_aligned data """

import os
import glob
import numpy as np
import pandas as pd
import PIL
import matplotlib.pyplot as plt

path_celeba_img = './data/processed/celebA_crop'
path_celeba_att = './data/raw/celebA_annotation/list_attr_celeba.txt'

##
""" loading attributes and checking image file names """
df_attr = pd.read_csv(path_celeba_att, sep='\s+', header=1, index_col=0)

img_names = os.listdir(path_celeba_img)
img_names = [img_name for img_name in img_names if img_name[-4:]=='.jpg']
img_names.sort()

assert df_attr.shape[0] == len(img_names), 'images number does not match attribute table'
num_img = df_attr.shape[0]


print(df_attr.head(3))
print(df_attr.tail(3))
print(img_names[:3])
print(img_names[-3:])

assert df_attr.shape[0] == len(img_names), 'images number does not match attribute table'
num_img = df_attr.shape[0]

##



##
