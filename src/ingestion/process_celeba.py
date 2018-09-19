"""
Download and extract celebA dataset

Modification of https://github.com/stanfordnlp/treelstm/blob/master/scripts/download.py
Downloads the following:
- Celeb-A dataset
- LSUN dataset
- MNIST dataset
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

path_data_raw = './data/raw'

parser = argparse.ArgumentParser(description='Download dataset.')
parser.add_argument('datasets', metavar='N', type=str, nargs='+', choices=['celebA', 'cifar', 'mnist'],
                   help='name of dataset to download [celebA, cifar, mnist]')

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    u = urllib.request.urlopen(url)
    f = open(filepath, 'wb')
    filesize = int(u.headers["Content-Length"])
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
            ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)


def reshape_celebA(path_to_data):
    from scipy import misc
    import numpy as np
    from PIL import Image
    files_read = []
    for root, subFolders, files in os.walk(path_to_data):
        print(root)
        print(subFolders)
        print(len(files))
        for f in files:
            if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'):
                files_read.append(os.path.join(root, f))
                # print(files_read[-1])
        print('one subdir done')
    # files = [f for f in os.listdir(path_to_data) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
    print('Done listing files')
    images = []
    for f in files_read:
        try:
            # im = misc.imread(f)
            im = Image.open(f)
            im = np.array(im)
            # print(im)
        except IOError:
            print('Could not read: %s' % f)
        if len(im.shape) == 2:
            im = np.expand_dims(im, -1)
        images.append(im)
    print('Done reading files')
    num_c = images[0].shape[-1]
    for i in range(len(images)):
        images[i] = misc.imresize(images[i], (64, 64, num_c))
        # if len(images[i].shape) == 3:
        #     images[i] = np.expand_dims(images[i], 0)
    data = np.stack(images, axis=0).astype(np.float32)
    np.save(os.path.join(path_to_data, 'celeb_64.npy'), data)

def download_celeb_a(dirpath):
    data_dir = 'celebA'
    if os.path.exists(os.path.join(dirpath, data_dir)):
        print('Found Celeb-A - skip')
        return
    url = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1&pv=1'
    filepath = download(url, dirpath)
    zip_dir = ''
    with zipfile.ZipFile(filepath) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(filepath)
    os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))
    reshape_celebA(os.path.join(dirpath, data_dir))

def _list_categories(tag):
    url = 'http://lsun.cs.princeton.edu/htbin/list.cgi?tag=' + tag
    f = urllib.request.urlopen(url)
    return json.loads(f.read())

def _download_lsun(out_dir, category, set_name, tag):
    url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
          '&category={category}&set={set_name}'.format(**locals())
    print(url)
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = os.path.join(out_dir, out_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)


def download_lsun(dirpath):
    data_dir = os.path.join(dirpath, 'lsun')
    if os.path.exists(data_dir):
        print('Found LSUN - skip')
        return
    else:
        os.mkdir(data_dir)

    tag = 'latest'
    # categories = _list_categories(tag)
    categories = ['bedroom']

    for category in categories:
        _download_lsun(data_dir, category, 'train', tag)
        _download_lsun(data_dir, category, 'val', tag)
    _download_lsun(data_dir, '', 'test', tag)


def _download_cifar(out_dir):
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    print(url)
    # if set_name == 'test':
    #     out_name = 'test_lmdb.zip'
    # else:
    #     out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())

    file_path = os.path.join(out_dir, 'cifar-10-python.tar.gz')
    if not os.path.exists(file_path):
        cmd = ['wget', url, '-P', out_dir]
        print('Downloading CIFAR')
        subprocess.call(cmd)
    # tfile = tarfile.TarFile(file_path)
    with tarfile.open(name=file_path, mode='r:gz') as tfile:
        tfile.extractall(path=out_dir)


def download_cifar(dirpath):
    data_dir = os.path.join(dirpath, 'cifar-10-batches-py')
    if os.path.exists(data_dir):
        print('Found CIFAR - skip')
        return
    _download_cifar(dirpath)


def download_mnist(dirpath):
    data_dir = os.path.join(dirpath, 'mnist')
    if os.path.exists(data_dir):
        print('Found MNIST - skip')
        return
    else:
        os.mkdir(data_dir)
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = (url_base+file_name).format(**locals())
        print(url)
        out_path = os.path.join(data_dir,file_name)
        cmd = ['curl', url, '-o', out_path]
        print('Downloading ', file_name)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', out_path]
        print('Decompressing ', file_name)
        subprocess.call(cmd)

def prepare_data_dir(path = path_data_raw):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    args = parser.parse_args()
    prepare_data_dir()

    if 'celebA' in args.datasets:
        download_celeb_a(path_data_raw)
    if 'cifar' in args.datasets:
        download_cifar(path_data_raw)
    # if 'lsun' in args.datasets:
    #     download_lsun('./data')
    if 'mnist' in args.datasets:
        download_mnist(path_data_raw)