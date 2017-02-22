"""
Utility data functions
MSCOCO downsampled to 64x64 pixels

Author: Gabriel Bernard
Created on: 2017-02-21
Updated on: 2017-02-21
"""

import os
import glob
import PIL.Image as Image

# Try to import cpickle
try:
    import _pickle as pickle
except ImportError:
    try:
        import cPickle as pickle
    except ImportError:
        import pickle as pickle


def verify_archive(dataset):
    """
    Verify that an archive of the datasets is present and
    download the MSCOCO dataset from source url if not.

    :param dataset: path to the archive of the dataset
    :return: True if present or downloaded, false otherwise
    """
    directory, file = os.path.split(dataset)
    status = False

    if directory == "" and not os.path.isfile(dataset):
        path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(path) or file == 'inpainting.tar.bz2':
            dataset = path
            status = True

    # If the archive is not present, download it
    if (not os.path.isfile(dataset)) and file == 'inpainting.tar.bz2':
        from six.moves import urllib
        origin = ('http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/inpainting.tar.bz2')
        print('Data not present, downloading from {0}'.format(origin))
        urllib.request.urlretrieve(origin, dataset)

        print('Data needs to be extracted with: tar xjvf inpainting.tar.bz2')
        status = True

    return status


def verify_dataset(directory):
    """
    Verify that a directory is present
    or is in ../data/inpainting.

    :param directory: directory
    :return: the path to the directory
    """

    if os.path.isdir(directory):
        return directory

    path = os.path.join(
        os.path.split(__file__)[0],
        "..",
        "data",
        "inpainting",
        directory
    )
    if os.path.isdir(path):
        return path

    return None


def crop_data(dataset_path, save_dir):
    """
    Crop a 32x32 pixel square in the middle of the data
    to feed to the neural network.

    :param directory: path to data
    """

    data_path = verify_dataset(dataset_path)
    if data_path is None:
        raise NotADirectoryError('{0} is not a directory and is not in ../data'.format(dataset_path))

    if verify_dataset(save_dir) is None:
        print("Creating save directory to {0}".format(save_dir))
        os.makedirs(save_dir)

    preserve_ratio = True
    img_size = (64, 64)

    data = glob.glob(data_path + "/*.jpg")

    for i, img_path in enumerate(data):
        img = Image.open(img_path)
        print(i, len(data), img_path)

        # if img.
