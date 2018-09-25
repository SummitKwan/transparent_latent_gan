""" train a simple cnn model for abstracting features from celebA_aligned data """

import os
import glob
import numpy as np
import pandas as pd
import PIL
import matplotlib.pyplot as plt

path_celeba_img = './data/raw/celebA'
path_celeba_att = './data/raw/celebA_annotation/list_attr_celeba.txt'

# loading attributes
df_attr = pd.read_csv(path_celeba_att, sep='\s+', header=1, index_col=0)

df_attr.head(3)

# load figures
# image_filenames = glob.glob(os.path.join(path_celeba_img, '*.jpg'))
# image_filenames.sort()

img_names = os.listdir(path_celeba_img)
img_names = [img_name for img_name in img_names if img_name[-4:]=='.jpg']
img_names.sort()

assert df_attr.shape[0] == len(img_names), 'images number does not match attribute table'
num_img = df_attr.shape[0]


##


def get_image_feature_pair(img_idx):
    img_name = img_names[img_idx]
    img = np.asarray(PIL.Image.open(os.path.join(path_celeba_img, img_name)))
    plt.imshow(img)
    plt.show()
    print
    # print(df_attr.ix[img_idx])
    print()


i_rand = np.random.randint(num_img)
get_image_feature_pair(i_rand)

##
pd.set_option('display.max_columns', 50)
i = np.random.randint(num_img)
print(image_filenames[i])
img = np.asarray(PIL.Image.open(image_filenames[i]))

cx=89
cy=121
assert img.shape == (218, 178, 3)
img_crop = img[cy - 64 : cy + 64, cx - 64 : cx + 64]

plt.imshow(img_crop)
plt.show()
df_attr[i:i+1]