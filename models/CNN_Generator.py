"""
Simple convolutionnal generator made with Lasagne

This was deeply inspired by the work of
Chin-Wei Huang https://chinweihuang.com/2017/02/22/hello-convnet/
to try and learn how to use Lasagne

Author: Gabriel Bernard
Updated on: 2017-03-25
"""

import theano
import theano.tensor as T
import lasagne
from lasagne.nonlinearities import tanh, rectify, sigmoid
import glob
import os
import PIL.Image as Image
import numpy as np

try:
    from models.utils import data_utils
except ImportError:
    from utils import data_utils


def cnn_encoder(cnn=None, input_shape=(None, 3,64,64),
                n_layers=None, list_of_nfilters=None,
                list_of_filter_sizes=None, list_of_strides=None,
                list_of_paddings=None, list_of_pool_sizes=None,
                list_of_pool_modes=None, list_of_activations=None):
    """
    Create a CNN network to encode the impots informations

    :param cnn: Convolutional neural net
    :param input_shape: tuple of size 4 (batch_size, features, height, width)
    :param n_layers: Number of layers
    :param list_of_nfilters: Number of filters for each layers
    :param list_of_filter_sizes: Size of the filters for each layers
    :param list_of_strides: Stride applied on each layer of the network
    :param list_of_paddings: Padding for each layers
    :param list_of_pool_sizes: Pool for each layers
    :param list_of_pool_modes: Pool modes for each layers
    :param list_of_activations: Activation function for each layers
    :return: The network
    """
    if cnn is None:
        cnn = lasagne.layers.InputLayer(
            shape=input_shape, input_var=None)

    if n_layers is not None:
        for i in range(n_layers):
            filters = list_of_nfilters[i]
            filter_size = list_of_filter_sizes[i]
            stride = list_of_strides[i]
            padding = list_of_paddings[i]
            pool_size = list_of_pool_sizes[i]
            pool_mode = list_of_pool_modes[i]
            activation = list_of_activations[i]

            cnn = lasagne.layers.Conv2DLayer(
                incoming=cnn, num_filters=filters,
                filter_size=filter_size, stride=stride, pad=padding,
                nonlinearity=activation
            )

            if pool_size is not None:
                lasagne.layers.Pool2DLayer(
                    cnn, pool_size=pool_size, mode=pool_mode
                )
    else:
        raise ValueError("A number of layers should be given in cnn_encoder")

    return cnn

def cnn_decoder(dcnn=None, n_layers=None,
                input_shape=None, list_of_nfilters=None,
                list_of_filter_sizes=None, list_of_strides=None,
                list_of_paddings=None, list_of_pool_sizes=None,
                list_of_pool_modes=None, list_of_activations=None):

    """

    :param dcnn: Deconvolutional neural network
    :param n_layers: Number of layers
    :param input_shape: Tuple of size 4
    :param list_of_nfilters: Number of filters for each layers
    :param list_of_filter_sizes: Size of filters for each layers
    :param list_of_strides: Strides for each layers
    :param list_of_paddings: Padding for each layers
    :param list_of_pool_sizes: Pool size for each layers
    :param list_of_pool_modes: Pool mode for each layers
    :param list_of_activations: Activation funciton for each layers
    :return: The network
    """
    if dcnn is None:
        dcnn = lasagne.layers.InputLayer(
            shape=input_shape, input_var=None)

    if n_layers is not None:

        for i in range(n_layers):
            filters = list_of_nfilters[i]
            filter_size = list_of_filter_sizes[i]
            stride = list_of_strides[i]
            padding = list_of_paddings[i]
            pool_size = list_of_pool_sizes[i]
            pool_mode = list_of_pool_modes[i]
            activation = list_of_activations[i]

            if pool_size is not None:
                dcnn = lasagne.layers.Upscale2DLayer(
                    dcnn, scale_factor=pool_size, mode=pool_mode
                )

            dcnn = lasagne.layers.Conv2DLayer(
                incoming=dcnn, num_filters=filters,
                filter_size=filter_size, stride=stride, pad=padding,
                nonlinearity=activation
            )
    else:
        raise ValueError("A number of layers shoud be given in cnn_decoder")

    return dcnn


def preload_batches(list_of_image, list_of_target, first, last):
    """
    Function that preloads some minibatches
    :param list_of_image: Path and name of all images
    :param list_of_target: Path and name of all targets
    :param first: First image to load
    :param last: Last image to load
    :return: All image and targets from first to last
    """
    preload_input_batch = data_utils.load_data(
        list_of_image[first:last], (64, 64)
    ).astype(theano.config.floatX)
    preload_target_batch = data_utils.load_data(
        list_of_target[first:last], (32, 32)
    ).astype(theano.config.floatX)

    return preload_input_batch, preload_target_batch


def main():
    batch_size = 128
    data_path = "/Users/Gabriel/PycharmProjects/Conditional_Image_Generation/data/input"
    list_of_image = glob.glob(data_path + "/train2014" + "/input_*.jpg")
    list_of_target = glob.glob(data_path + "/train2014" + "/target_*.jpg")
    list_of_image = list_of_image[0:1000]
    list_of_target = list_of_target[0:1000]

    assert len(list_of_image) is not 0
    assert len(list_of_image) == len(list_of_target)

    n_batch = len(list_of_image) // batch_size

    print("Creating Encoder")
    cnn = cnn_encoder(cnn=None,
                    n_layers=5,
                    list_of_nfilters=[64, 64, 64, 64, 64],
                    list_of_filter_sizes=[3, 3, 3, 3, 3],
                    list_of_strides=[1, 1, 2, 2, 3],
                    list_of_paddings=[0, 0, 0, 0, 0],
                    list_of_pool_sizes=[None, None, None, None, None],
                    list_of_pool_modes=[None, None, None, None, None],
                    list_of_activations=[tanh, rectify, rectify, rectify, tanh]
                    )

    cnn = lasagne.layers.DenseLayer(incoming=cnn, num_units=1024, nonlinearity=tanh)

    print("Creating unification layer")
    cnn = lasagne.layers.ReshapeLayer(
        cnn,
        [[0]] + [64, 4, 4]
    )

    print("Creating Decoder")
    cnn = cnn_decoder(dcnn=cnn,
                n_layers=5,
                input_shape=None,
                list_of_nfilters=[64, 64, 64, 64, 3],
                list_of_filter_sizes=[5, 4, 3, 3, 3],
                list_of_strides=[1, 1, 1, 1, 1],
                list_of_paddings=[4, 3, 2, 1, 1],
                list_of_pool_sizes=[2, None, 2, None, None],
                list_of_pool_modes=['dilate', None, 'repeat', None, None],
                list_of_activations=[tanh, rectify, rectify, rectify, sigmoid]
    )

    print("Building Model")
    x = T.tensor4('x')
    target = T.tensor4('target')

    x = x.reshape((batch_size, 3, 64, 64))

    pred = lasagne.layers.get_output(cnn, x)
    loss = T.mean(lasagne.objectives.squared_error(pred, target))
    params = lasagne.layers.get_all_params(cnn)
    updates = lasagne.updates.adam(loss, params, 0.001)

    # print("Building the model")
    train_func = theano.function(
        [x, target],
        loss,
        updates=updates
    )

    predict = theano.function([x], pred)

    save_dir = "./tmp/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_epoch = 10
    n_of_batch_to_preload = 10
    n_of_preloaded_batch = n_of_batch_to_preload * batch_size

    # with np.load(save_dir + 'model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(cnn, param_values)

    print("Beginning Training")
    for e in range(n_epoch):

        for j in range(n_batch):
            n = -1
            if j % n_of_preloaded_batch == 0:
                n += 1
                first = n_of_preloaded_batch * n
                last = n_of_preloaded_batch * (n + 1)

                if last > len(list_of_image):
                    last = len(list_of_image)
                if first > len(list_of_image):
                    raise ValueError("First to high!!!!!!!!!!")

                preload_input_batch, preload_target_batch = preload_batches(list_of_image, list_of_target, first, last)

            first = (batch_size * j) % n_of_preloaded_batch
            last = first + batch_size

            if last > len(preload_input_batch):
                last = len(preload_input_batch)

            in_data = preload_input_batch[first:last]
            in_target = preload_target_batch[first:last]

            loss = train_func(in_data, in_target)

            if j % 10 == 0:
                print("Epoch: {0}/{1}, Batch: {2}/{3}, Loss: {4}".format(e, n_epoch, j, n_batch, loss))
                np.savez(save_dir + 'model.npz', *lasagne.layers.get_all_param_values(cnn))

            if (j == n_batch - 1) & (((j+1) * batch_size) < len(list_of_image)):
                first = (j+1) * batch_size
                last = len(list_of_image)
                in_data, in_target = preload_batches(list_of_image, list_of_target, first, last)
                loss = train_func(in_data, in_target)

        print(e, loss)
    in_data, target_batch = preload_batches(list_of_image, list_of_target, 0, 10)
    out = predict(in_data)
    for i in range(out.shape[0]):
        img = out[i] * 255
        img = img.reshape(32, 32, 3)
        img = np.uint8(img)
        img = Image.fromarray(img)
        img.save(save_dir + str(i) + ".jpg")


if __name__ == "__main__":
    main()
