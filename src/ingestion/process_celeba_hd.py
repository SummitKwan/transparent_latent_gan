"""
Download and extract celebA dataset (original version, un-aligned)

Note: to run this script, first make sure the datafile is manually downloaded and stored at './data/raw/celebA_wild_7z'
celebA, orignial (non-aligned) version of data can be downloaded from: 
https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg
in the local hard disk, it should be
./data/raw/celebA_wild_7z/img_celeba.7z.001
                          img_celeba.7z.002
                          ...
                          img_celeba.7z.014

celebA annotations should be manually downloaded and stored at './data/raw/celebA_annotation'
celabA annotations can be downloaded at:
https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?usp=drive_open
in the local hard disk, it should be
./data/raw/celebA_annotation/identity_CelebA.txt
                             list_attr_celeba.txt
                             list_bbox_celeba.txt
                             list_landmarks_align_celeba.txt
                             list_landmarks_celba.txt


celebA HQ delta should be manually downloaded and stored at './data/raw/celebA_hq_delta'
it can be downloaded at:
https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs
in the local hard disk, it should be
./data/raw/celebA_hq_deltas/deltas00000.zip
                            deltas01000.zip
                            ...
                            deltas29000.zip

"""

import os
import sys
import gzip
import json
import shutil
import zipfile
import tarfile
import argparse
import subprocess
from six.moves import urllib


""" process celebA in the wild """
path_celebA_alinged = '.data/raw/celebA'
path_celebA_wild_7z = './data/raw/celebA_wild_7z'
name_file_first = 'img_celeba.7z.001'
name_file_combined = 'img_celeba.7z'
path_celebA_wild_extracted = './data/raw/celebA_wild'

path_celebA_wild_7z_file_to_extract = os.path.join(path_celebA_wild_7z, name_file_first)

if not os.path.exists(path_celebA_wild_extracted):
    os.mkdir(path_celebA_wild_extracted)

if os.path.exists(path_celebA_wild_7z_file_to_extract):
    os.system('7z x {} -tiso.split -o{}'.format(path_celebA_wild_7z_file_to_extract, path_celebA_wild_extracted))
    os.system('7z x {} -o{}'.format(os.path.join(path_celebA_wild_extracted, name_file_combined),
                                    path_celebA_wild_extracted))
else:
    raise Exception('data file does not exist for extraction ./data/raw/celebA_wild_7z//img_celeba.7z.001')


""" process celebA HQ delta """
path_celebA_hq = './data/raw/celebA_hq_deltas_zip'


""" generate celeb HQ data """
# not used heres

# run the following in terminal to generate tf_record version of data
# ~/ana*/envs/ten*p36/bin/python ./src/ingestion/dataset_tool_modify.py create_celeba ./data/processed/celeba ./data/raw/celebA


