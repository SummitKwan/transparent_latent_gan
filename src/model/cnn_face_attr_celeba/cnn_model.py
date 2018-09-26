""" train and test for a convolutional neural network for predicting face attrubute for celebA """

import os
import glob
import numpy as np
import pandas as pd
import PIL
import matplotlib.pyplot as plt
import tensorflow as tf

path_celeba_img = './data/processed/celebA_crop'
path_celeba_att = './data/raw/celebA_annotation/list_attr_celeba.txt'

##
""" data loader """
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

num_feature = df_attr.shape[1]

##
""" transfer learning in keras """

import keras
import numpy as np
import keras.applications
import keras.layers as layers

# Load the pretrained model


base_model = keras.applications.mobilenet.MobileNet(include_top=False, input_shape=(128,128,3),
                                                      alpha=1, depth_multiplier=1,
                                                      dropout=0.001, weights="imagenet",
                                                      input_tensor=None, pooling=None)

base_model.summary()

##
fc0 = base_model.output
fc0_pool = layers.GlobalAveragePooling2D(data_format='channels_last', name='fc0_pool')(fc0)
fc1 = layers.Dense(256, activation='relu', name='fc1_dense')(fc0_pool)
fc2 = layers.Dense(num_feature, activation='tanh', name='fc2_dense')(fc1)
# x1= layers.Dense(32)(x0)

model = keras.models.Model(inputs=base_model.input, outputs=fc2)
model.compile(optimizer='sgd', loss='mean_squared_error')

for layer in base_model.layers:
    layer.trainable = False


model.summary()

##
""" get some data and train """
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt


# load an image in PIL format
original = load_img(os.path.join(path_celeba_img, img_names[0]))
print('PIL image size', original.size)
plt.imshow(original)
plt.show()

# convert the PIL image to a numpy array
# IN PIL - image is in (width, height, channel)
# In Numpy - image is in (height, width, channel)
numpy_image = img_to_array(original)
plt.imshow(np.uint8(numpy_image))
plt.show()
print('numpy array size', numpy_image.shape)

# Convert the image / images into batch format
# expand_dims will add an extra dimension to the data at a particular axis
# We want the input matrix to the network to be of the form (batchsize, height, width, channels)
# Thus we add the extra dimension to the axis 0.
image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)
plt.imshow(np.uint8(image_batch[0]))
plt.show()

# preprocess image
image_batch = keras.applications.mobilenet.preprocess_input(image_batch.copy())



##
base_model_full = keras.applications.mobilenet.MobileNet(input_shape=(128,128,3),
                                                      alpha=1, depth_multiplier=1,
                                                      dropout=0.001, weights="imagenet",
                                                      input_tensor=None, pooling=None)
base_model_full.summary()


