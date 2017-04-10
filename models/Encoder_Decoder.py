import argparse
import numpy as np
import theano.tensor as T
import theano
import lasagne
from lasagne.layers import Conv2DLayer, DenseLayer, batch_norm, InputLayer, ReshapeLayer, Deconv2DLayer
from lasagne.nonlinearities import sigmoid
from lasagne.objectives import binary_crossentropy
import os
import glob
import time

try:
    from models.utils import data_utils
except ImportError:
    from utils import data_utils


def image_encoder(input_var=None):

    # Output size of convolution formula:
    # o = (i + 2p - k) / s + 1
    # Where o is the output size, i the input, p
    # the padding, k the kernel size and s the stride

    tanh = lasagne.nonlinearities.tanh
    net = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
    # 128 units of 32 x 32
    net = batch_norm(Conv2DLayer(net, 128, 2, stride=2))
    # 256 units of 16 x 16
    net = batch_norm(Conv2DLayer(net, 256, 2, stride=2))
    # 512 units of 8 x 8
    net = batch_norm(Conv2DLayer(net, 512, 2, stride=2))
    # 1024 units of 4 x 4
    net = batch_norm(Conv2DLayer(net, 1024, 2, stride=2))
    # Fully connected layer
    net = DenseLayer(net, 100, nonlinearity=tanh)

    print("Image encoder output shape: ", net.output_shape)
    return net


def image_decoder(input_var=None):

    # Output size of deconvolution formula:
    # o = s(i - 1) + a + k - 2p
    # Where o is the output size, i the input, p
    # the padding, k the kernel size and s the stride

    tanh = lasagne.nonlinearities.tanh
    net = InputLayer(shape=(None, 100), input_var=input_var)
    # Project
    net = batch_norm(DenseLayer(net, 1024 * 4 * 4, nonlinearity=tanh))
    # Reshape
    net = ReshapeLayer(net, ([0], 1024, 4, 4))
    # 512 units of 8 x 8
    net = batch_norm(Deconv2DLayer(net, 512, 2, stride=2))
    # 256 units of 16 x 16
    net = batch_norm(Deconv2DLayer(net, 256, 9))
    # 128 units of 32 x 32
    net = batch_norm(Deconv2DLayer(net, 128, 2, stride=2))
    # 3 units of 64 x 64 (rgb image)
    net = Deconv2DLayer(net, 3, 2, stride=2)

    print("Image decoder output shape ", net.output_shape)
    return net


def train(epoch, batch_size, learning_rate, datapath, savepath):

    x = T.tensor4('x')
    x = x.reshape((batch_size, 3, 64, 64))
    code = T.matrix('target')

    print("Building model")
    encoder = image_encoder(x)
    decoder = image_decoder(code)

    encode = lasagne.layers.get_output(encoder, x)
    decode = lasagne.layers.get_output(decoder, encode)
    loss = lasagne.objectives.squared_error(decode, x).mean()

    params = lasagne.layers.get_all_params(encoder)
    params = lasagne.layers.get_all_params(decoder)
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

    train_fn = theano.function(
        [x, x],
        loss,
        updates=updates
    )

    for e in epoch:
        for b in data_utils.minibatch_iterator():
            inp, tar = b



def pars_args():
    arguments = argparse.ArgumentParser()
    arguments.add_argument(
        '-dp', '--DataPath', type=str,
        help="Complete path to dataset"
    )
    arguments.add_argument(
        '-sp', '--SavePath', type=str,
        help="Complete path to directory where to store computation"
    )
    arguments.add_argument(
        '-bs', "--BatchSize", type=int,
        help="Size of the minibatch to train"
    )
    arguments.add_argument(
        "-ne", "--epochs", type=int, default=200,
        help="Number of step for training"
    )
    arguments.add_argument(
        '-t', '--train', type=int, default=1,
        help="Define if the GAN must be trained"
    )
    arguments.add_argument(
        '-lr', '--LearningRate', type=float, default=2e-4,
        help="Defines the learning rate of the GAN network"
    )
    arguments.add_argument(
        '-lp', "--LoadPath", type=str, default=None,
        help="Complete path to directory where to load stored computation"
    )
    return arguments.parse_args()


def main(args):
    train(
        epoch=args.epochs,
        batch_size=args.BatchSize,
        learning_rate=args.LearningRate,
        datapath=args.DataPath,
        savepath=args.SavePath
    )
    print("End")


if __name__ == '__main__':
    main(pars_args())
