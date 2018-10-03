"""
script to crop celebA dataset and save into new folder
"""

import os
import glob
import numpy as np
import pandas as pd
import PIL
import PIL.Image
import h5py

path_celeba_img = './data/raw/celebA'
path_celeba_att = './data/raw/celebA_annotation/list_attr_celeba.txt'
path_celeba_crop = './data/processed/celebA_crop'
path_celeba_crop_h5 = './data/processed/celebA_crop_h5'
filename_h5 = 'celebA_crop.h5'

if not os.path.exists(path_celeba_crop):
    os.mkdir(path_celeba_crop)

##
""" image crop """
def img_crop(img, cx=89, cy=121, w=128, h=128):
    """
    crop images based on center and width, height

    :param img: image data, numpy array, shape = [height, width, RGB]
    :param cx:  center pixel, x
    :param cy:  center pixel, y
    :param w:   width, even number
    :param h:   height, even number
    :return: img_crop
    """
    img_cropped = img[cy-h//2: cy+h//2, cx-w//2: cx+w//2]
    return img_cropped

##
""" get image and attribute data """

# loading attributes
df_attr = pd.read_csv(path_celeba_att, sep='\s+', header=1, index_col=0)

img_names = os.listdir(path_celeba_img)
img_names = [img_name for img_name in img_names if img_name[-4:]=='.jpg']
img_names.sort()

assert df_attr.shape[0] == len(img_names), 'images number does not match attribute table'
num_img = df_attr.shape[0]

##
""" save cropped image to harddisk """

for i_img in range(num_img):
    if i_img%100 == 0:
        print('{}/{}'.format(i_img, num_img))
    img_name = img_names[i_img]
    img = np.asarray(PIL.Image.open(os.path.join(path_celeba_img, img_name)))
    img = img_crop(img)
    PIL.Image.fromarray(img).save(os.path.join(path_celeba_crop, img_name))
print('finished {} images, saved in {}'.format(num_img, path_celeba_crop))

##
""" test cropping and whether image and label matches """
yn_interactive_test = False
yn_time_img_loading = False

if yn_interactive_test:
    import matplotlib.pyplot as plt

    i = np.random.randint(num_img)
    print(df_attr.ix[i])
    print("image file name: {}".format(img_names[i]) )

    img = np.asarray(PIL.Image.open(os.path.join(path_celeba_img, img_names[i])))

    plt.imshow(img_crop(img))
    plt.show()


##
if yn_time_img_loading:
    import time
    num_times = 1000
    tic = time.time()
    for i_time in range(num_times):
        i = np.random.randint(num_img)
        np.asarray(PIL.Image.open(os.path.join(path_celeba_img, img_names[i])))
    toc = time.time()
    print((toc-tic)/num_times)

##
yn_use_h5 = False

def fun_get_img(file_img):
    img = np.asarray(PIL.Image.open(os.path.join(path_celeba_img, file_img)))
    return img_crop(img)

if yn_use_h5:

    filepath_h5 = os.path.join(path_celeba_crop_h5, filename_h5)
    if not os.path.exists(path_celeba_crop_h5):
        os.mkdir(path_celeba_crop_h5)

    """ crop data and save to h5 """
    def save_to_h5_img(filepath_h5=filepath_h5, list_img_file=tuple(), fun_get_img=fun_get_img, dataset_name='img'):
        """
        save the images as hdf5 format
        """
        if os.path.exists(filepath_h5):
            print('h5 file exists, please delete it or give another name. if you want to overwrite, type "overwrite"')
            key_in = input()
            if key_in == 'overwrite':
                print('overwrite file {}'.format(filepath_h5))
            else:
                raise Exception()
        if len(list_img_file) == 0:
            raise Exception('no input data')

        img = fun_get_img(list_img_file[0])
        print(img.shape)

        plt.imshow(img);
        plt.show()

        with h5py.File(filepath_h5, 'a') as hf:
            hf.create_dataset(dataset_name, data=img)

    save_to_h5_img(list_img_file=[img_names[0]])

    ##
    """ test_read_h5 """
    tic = time.time()
    for i in range(1000):
        with h5py.File(filepath_h5, 'r') as hf:
            img = hf['img'][:]
    toc = time.time()
    print((toc-tic)/1000)

    plt.imshow(img);
    plt.show()

