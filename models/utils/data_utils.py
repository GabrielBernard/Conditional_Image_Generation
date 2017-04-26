"""
Utility data functions
MSCOCO downsampled to 64x64 pixels

Author: Gabriel Bernard
Updated on: 2017-03-07
"""

import os
import glob
import gc
import numpy as np
import PIL.Image as Image

# Try to import cpickle
import six.moves.cPickle as pickle
# import _pickle as pickle


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
    or is in ../../data/inpainting
    (default directory relative to this file).

    :param directory: directory
    :return: the path to the directory
    """

    if os.path.isdir(directory):
        return directory

    path = os.path.join(
        os.path.split(__file__)[0],
        "..",
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
    to feed to the neural network. Save the images created
    in the save_dir for future use and make a pickle file
    containing image id, cropped image and cropped pattern
    (to use when training)

    :param dataset_path: path to data
    :param save_dir: path where to save croped images
    """

    # Verification of existance of dataset path
    data_path = verify_dataset(dataset_path)
    if data_path is None:
        raise NotADirectoryError('{0} is not a directory and is not in ../data'.format(dataset_path))

    # Verification of save dir path
    if verify_dataset(save_dir) is None:
        print("Creating save directory to {0}".format(save_dir))
        os.makedirs(save_dir)  # Creates dir

    # Retrieve all jpg file names in dataset_path
    data = glob.glob(data_path + "/*.jpg")

    # Initialization
    grayscale = 0
    dic = {}
    index = 0
    # Iteration over all jpg files in dataset_path
    for i, img_path in enumerate(data):

        # Opening image
        img = Image.open(img_path)

        # Printing progress to console every 1000 images
        if (i % 1000) == 0:
            print(i, len(data), img_path)

        # Creates a numpy array from image
        array = np.array(img)

        # Retrieve name of image
        img_name = os.path.basename(img_path)

        # Stripping .jpg at end of image name
        cap_id = img_name[:-4]

        # Finding center coordinate of image
        center = (
            int(np.floor(array.shape[0] / 2.)),
            int(np.floor(array.shape[1] / 2.))
        )

        # Finding if array is grayscale
        if len(array.shape) == 3:
            # If not, cropping center
            input = np.copy(array)
            input[
                center[0] - 16: center[0] + 16,
                center[1] - 16: center[1] + 16, :
            ] = 0
            # Registering center to special variable
            target = array[
                center[0] - 16:center[0] + 16,
                center[1] - 16:center[1] + 16, :
            ]
            # For each non grayscale image
            input_img = Image.fromarray(input)
            target_img = Image.fromarray(target)
            # Save cropped image and target to save_dir
            input_img.save(os.path.join(save_dir, "input_" + img_name))
            target_img.save(os.path.join(save_dir, "target_" + img_name))
            img.save(os.path.join(save_dir, 'img_' + img_name))
            # Update dictionnary with cap_id, input and target
            dic.update({index: cap_id})
            index += 1
        else:
            # If grayscale, printing which is
            # print(i, "grayscale")
            grayscale += 1
        # Force garbage collection after 10000 cropping
        # to not overflow virtual memorry
        if (i % 10000) == 0:
            gc.collect()

    # Print end of cropping
    print("End cropping")

    # Printing how many grayscale images where found
    print("There were {0} grayscale images.".format(grayscale))
    # Saving dictionnary to pickle file
    pickle_save_path = os.path.join(save_dir, "data.pkl")
    print("registering pickle file")
    pickle.dump(dic, open(pickle_save_path, 'wb'))
    print("End of cropping with last index: ", index)


def load_data(list_of_images, size):
    """
    This function loads the list_of_images
    knowing their size (row, column).

    :param list_of_images: List that contains all the data names of a mini batch
    :param size: Tuple that contains the size of the images
    :return: Numpy array containing the batch of data in the form
        [ batch, channels, height, width ]
    """
    ret = np.zeros([len(list_of_images), size[0], size[1], 3])

    for i, file in enumerate(list_of_images):
        # Load image
        img = Image.open(file)
        img = np.asarray(img, dtype='float32')
        # img = img.transpose((2, 0, 1))
        # img = img.reshape(3, size[0], size[1])
        ret[i] = img

    return ret.transpose((0, 3, 1, 2)) / 127.5 - 1


def load_data_to_ram(length, dic, prefixes, data_path, size=[(64, 64), (64, 64)]):
    l = min(length, len(dic))
    size1 = size[0]
    size2 = size[1]
    input_array = np.zeros((l, size1[0], size1[1], 3))
    target_array = np.zeros((l, size2[0], size2[1], 3))
    input = data_path + prefixes[0]
    target = data_path + prefixes[1]
    j = 0
    for i in dic:
        name = dic[i] + '.jpg'
        img = Image.open(input + name)
        img = np.asarray(img, dtype='float32')
        # img = img.transpose(2, 0, 1)
        # img = img.reshape(3, size1[0], size1[1])
        # input_array[j] = img / 255
        input_array[j] = img

        img = Image.open(target + name)
        img = np.asarray(img, dtype='float32')
        # img = img.reshape(3, size2[0], size2[1])
        # target_array[j] = img / 255
        target_array[j] = img
        j += 1

        if (j == l) | (i == len(dic) - 1):
            input_array = input_array.transpose((0, 3, 1, 2)) / 127.5 - 1
            target_array = target_array.transpose((0, 3, 1, 2)) / 127.5 - 1
            yield input_array, target_array
            l = np.floor(min(length, len(dic) - i)).astype(np.int)
            input_array = np.zeros((l, size1[0], size1[1], 3))
            target_array = np.zeros((l, size2[0], size2[1], 3))
            # input = data_path + prefixes[0]
            # target = data_path + prefixes[1]
            j = 0


def minibatch_iterator(x, y, batch_size):
    """
    Iterator on a minibatch.

    :param x: Input data to fetch
    :param y: Target data to fetch
    :param batch_size: Size of a batch
    :return: Two arrays containing the input and target
    """
    assert len(x) == len(y)
    i = None
    for i in range(0, len(x) - batch_size + 1, batch_size):
        batch = slice(i, i + batch_size)
        yield x[batch], y[batch]
    # Make sure that all the dataset is passed
    # even if it is less then a full batch_size
    if i is None:
        i = 0
    # Fetch the last data from the dataset
    if len(x) - batch_size < 0:
        batch = slice(i, len(x))
        yield x[batch], y[batch]

    if i + batch_size < len(x):
        batch = slice(i + batch_size, len(x) - 1)
        yield x[batch], y[batch]
