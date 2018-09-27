""" predict_feature labels of synthetic_images """

import os
import glob
import numpy as np
import PIL.Image
import h5py
import src.model.cnn_face_attr_celeba as cnn_face

# path to model generated results
path_gan_sample_img = './asset_results/pggan_celeba_sample_jpg/'
file_pattern_x = 'sample_*.jpg'
file_pattern_z = 'sample_*_z.npy'

# get the list of image_names
list_pathfile_x = glob.glob(os.path.join(path_gan_sample_img, file_pattern_x))
list_pathfile_z = glob.glob(os.path.join(path_gan_sample_img, file_pattern_z))
list_pathfile_x.sort()
list_pathfile_z.sort()

assert len(list_pathfile_x) == len(list_pathfile_z), 'num_image does not match num_z'

##
""" load model for prediction """
model = cnn_face.create_cnn_model()
model.load_weights(cnn_face.get_list_model_save()[-1])

i_counter = 0
for pathfile_x in list_pathfile_x[:64]:
    img = np.asarray(PIL.Image.open(pathfile_x))


