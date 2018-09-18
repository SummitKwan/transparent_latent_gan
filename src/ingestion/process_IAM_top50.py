""" process the iam-handwriting-top50 dataset """
##
import os
import warnings
import zipfile

##
""" unzip file """
# assume you are running from the project base

path_zipfile_iam_top50 = './data/raw/iam-handwriting-top50.zip'
path_extract_iam_top50 = './data/raw/'
if os.path.exists(path_zipfile_iam_top50):
    zipfile.ZipFile(path_zipfile_iam_top50, 'r') \
        .extractall(path_extract_iam_top50)
else:
    warnings.warn('please sdownload the iam-top50 dataset zip file from Kaggle and put it under ./data/raw')

path_zipfile_data_subset = './data/raw/data_subset.zip'
path_extract_data_subset = './data/raw/data_subset'
if not os.path.exists(path_extract_data_subset):
    os.mkdir(path_extract_data_subset)


zipfile.ZipFile(path_zipfile_data_subset, 'r') \
    .extractall(path_extract_iam_top50)
