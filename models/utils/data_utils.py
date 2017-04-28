"""
Utility data module
MSCOCO downsampled to 64x64 pixels

Author: Gabriel Bernard
Updated on: 2017-04-28
"""

import os
import glob
import gc
import numpy as np
import PIL.Image as Image

import six.moves.cPickle as pickle


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
    :return: Numpy array containing the normalized batch of data in the form
        [ batch, channels, height, width ]
    """

    # Initialize return array
    ret = np.zeros([len(list_of_images), size[0], size[1], 3])

    # Fetch the data on disk
    for i, file in enumerate(list_of_images):
        # Load image
        img = Image.open(file)
        img = np.asarray(img, dtype='float32')
        ret[i] = img

    # Normalize and return data
    return ret.transpose((0, 3, 1, 2)) / 127.5 - 1


def load_data_to_ram(length, dic, prefixes, data_path, size=[(64, 64), (64, 64)]):
    """
    Generator that loads multiple batches of inputs and targets
    on ram and yields them until it has passed all the dataset (dic).

    :param length: Number of batches to load on ram
    :param dic: Dictionnary containing the names of data to fetch
    :param prefixes: Prefixes of the data to fetch in the form [inputs_pref, targets_pref]
    :param data_path: Path from where to load the data
    :param size: Array containing two tuples with the height and width of the inputs and targets

    :yield input_array, target_array: Arrays containing the inputs and targets
          for the training, in the form [batch, channels, height, width]
    """

    # Chosing the length in order to not overflow the dictionnary
    l = min(length, len(dic))
    # Fetching the sizes
    size1 = size[0]
    size2 = size[1]
    # Initialization of the input_array and target_array
    input_array = np.zeros((l, size1[0], size1[1], 3))
    target_array = np.zeros((l, size2[0], size2[1], 3))
    # Concatanating the datapath with the names' prefixes
    input = data_path + prefixes[0]
    target = data_path + prefixes[1]
    # Looping over the dictionnary's keys
    j = 0
    for i in dic:
        # Getting the mane of the image to load
        name = dic[i] + '.jpg'
        # Loading the input image
        img = Image.open(input + name)
        # Creating a numpy array of the image
        img = np.asarray(img, dtype='float32')
        # Registering the image in the input_array
        input_array[j] = img

        # Same as before but for the target image
        img = Image.open(target + name)
        img = np.asarray(img, dtype='float32')
        # Registering in the target_array
        target_array[j] = img
        # Counting the amount of images loaded
        j += 1
        # If j is the asked length or all the dictionnary was loaded
        if (j == l) | (i == len(dic) - 1):
            # Normalizing the data
            input_array = input_array.transpose((0, 3, 1, 2)) / 127.5 - 1
            target_array = target_array.transpose((0, 3, 1, 2)) / 127.5 - 1
            # yield everything
            yield input_array, target_array
            # Computing the next iteration's length
            l = np.floor(min(length, len(dic) - i)).astype(np.int)
            # Re-initializing the input and target array and the count
            input_array = np.zeros((l, size1[0], size1[1], 3))
            target_array = np.zeros((l, size2[0], size2[1], 3))
            j = 0


def minibatch_iterator(x, y, batch_size):
    """
    Iterator on a minibatch.

    :param x: Input data to array in the form [datas, channels, heigh, width]
    :param y: Target data to array in the form [datas, channels, height, width]
    :param batch_size: Size of a batch
    :return: Two arrays containing the input and target
    """

    # Making sure that there is the same amout of inputs x and targets y
    assert len(x) == len(y)
    # In case there is less data then the batch_size in x and y
    i = None
    # Fetching the data from the x and y arrays
    for i in range(0, len(x) - batch_size + 1, batch_size):
        batch = slice(i, i + batch_size)
        yield x[batch], y[batch]

    # Make sure that all the dataset is passed
    # even if x and y have less then a full batch_size
    if i is None:
        i = 0
    # Fetch all data from x and y if there is less then a full batch_size
    if len(x) - batch_size < 0:
        batch = slice(i, len(x))
        yield x[batch], y[batch]
    # Fetch remaining data if x and y's length is are factors of the batch_size
    if i + batch_size < len(x):
        batch = slice(i + batch_size, len(x) - 1)
        yield x[batch], y[batch]
