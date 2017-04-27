import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, Deconv2DLayer, BatchNormLayer, ReshapeLayer
from lasagne.nonlinearities import LeakyRectify, rectify, sigmoid, tanh
import logging
# import _pickle as pickle
import six.moves.cPickle as pickle
import argparse

try:
    from models.utils import data_utils
except ImportError:
    from utils import data_utils


def add_noize(input):
    center = (
        int(np.floor(input.shape[2] / 2.)),
        int(np.floor(input.shape[3] / 2.))
    )
    for i in range(input.shape[0]):
        input[i, :,
        center[0] - 16: center[0] + 16,
        center[1] - 16: center[1] + 16] = (np.random.random((3, 32, 32)) + 1) / 100

    return input


def discriminator(input_var=None):
    """
    Function that build the discriminator
    :param input_var: Input variable that goes in Lasagne.layers.InputLayer
    """

    # Output size of convolution formula:
    # o = (i + 2p - k) / s + 1
    # Where o is the output size, i the input, p
    # the padding, k the kernel size and s the stride

    lrelu = LeakyRectify(0.2)
    net = InputLayer((None, 3, 32, 32), input_var=input_var)
    # 128 units of 16 x 16
    net = BatchNormLayer(Conv2DLayer(net, 128, 2, stride=2, nonlinearity=lrelu))
    # 256 units of 8 x 8
    net = BatchNormLayer(Conv2DLayer(net, 256, 2, stride=2, nonlinearity=lrelu))
    # 512 units of 4 x 4
    net = BatchNormLayer(Conv2DLayer(net, 512, 2, stride=2, nonlinearity=lrelu))
    # 512 units of 8 x 8
    # net = batch_norm(Conv2DLayer(net, 512, 3, pad=1, nonlinearity=lrelu))
    # 1024 units of 4 x 4
    # net = batch_norm(Conv2DLayer(net, 1024, 2, stride=2, nonlinearity=lrelu))
    # Fully connected layers
    net = BatchNormLayer(DenseLayer(net, 512, nonlinearity=lrelu))
    net = DenseLayer(net, 1, nonlinearity=sigmoid)

    logging.info("Discriminator output shape : {}".format( net.output_shape))

    return net

def discriminator_op(input_var=None):
    Normal = lasagne.init.Normal
    lrelu = lasagne.nonlinearities.LeakyRectify(0.2)
    dis = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
    dis = lasagne.layers.GaussianNoiseLayer(dis, sigma=0.2)
    dis = BatchNormLayer(Conv2DLayer(dis, num_filters=96, filter_size=(3, 3), pad=1, W=Normal(0.05), nonlinearity=lrelu))
    dis = BatchNormLayer(Conv2DLayer(dis, num_filters=96, filter_size=(3, 3), pad=1, W=Normal(0.05), nonlinearity=lrelu))
    dis = (lasagne.layers.DropoutLayer(dis, p=0.5))
    dis = BatchNormLayer(Conv2DLayer(dis, num_filters=192, filter_size=(3, 3), pad=1, W=Normal(0.05), nonlinearity=lrelu))
    dis = BatchNormLayer(Conv2DLayer(dis, num_filters=192, filter_size=(3, 3), pad=1, W=Normal(0.05), nonlinearity=lrelu))
    dis = BatchNormLayer(Conv2DLayer(dis, num_filters=192, filter_size=(3, 3), pad=1, W=Normal(0.05), stride=2, nonlinearity=lrelu))
    dis = lasagne.layers.dropout(dis, p=0.5)
    dis = BatchNormLayer(Conv2DLayer(dis, num_filters=192, filter_size=(3, 3), pad=0, W=Normal(0.05), nonlinearity=lrelu))
    dis = BatchNormLayer(lasagne.layers.NINLayer(dis, num_units=192, W=Normal(0.05), nonlinearity=lrelu))
    dis = BatchNormLayer(lasagne.layers.NINLayer(dis, num_units=192, W=Normal(0.05), nonlinearity=lrelu))
    dis = lasagne.layers.GlobalPoolLayer(dis)
    dis = DenseLayer(dis, num_units=1, W=Normal(0.05), nonlinearity=sigmoid)

    return dis


def main(args):
    args.LoggingPath = os.path.expandvars(args.LoggingPath)
    print(args)
    post = 'log_' + time.strftime('%m_%d_%Y_%H_%M_%S') + '_gan3.log'

    if not os.path.isdir(args.LoggingPath):
        os.mkdir(args.LoggingPath)

    batch_size = args.BatchSize
    dic = pickle.load(open(args.DataPath + '/data.pkl', 'rb'))
    prefixes = ['/input_', '/target_']

    x = T.tensor4('x')

    # y = T.tensor4('y')
    dis = discriminator(x)

    prediction = lasagne.layers.get_output(dis)
    loss = lasagne.objectives.binary_crossentropy(T.mean(prediction), 1).mean()

    params = lasagne.layers.get_all_params(dis, trainable=True)
    updates = lasagne.updates.adam(
        loss, params, learning_rate=0.02
    )

    train_fn = theano.function(
        [x],
        loss,
        updates=updates
    )

    gen_fn = theano.function(
        [x],
        prediction,
    )

    for epoch in range(100):
        loss = 0
        it = 1
        for data in data_utils.load_data_to_ram(
                length=10 * batch_size,
                dic=dic,
                prefixes=prefixes,
                data_path=args.DataPath,
                size=[(64, 64), (32, 32)]
        ):
            for batch in data_utils.minibatch_iterator(x=data[0], y=data[1], batch_size=batch_size):
                inputs, targets = batch
                inputs = add_noize(inputs)
                inputs = inputs.astype(np.float32)
                targets = targets.astype(np.float32)
                loss += train_fn(targets)

                pred = gen_fn(targets)

                print(loss/it)
                it += 1
                # pred = gen_fn(inputs)

    print("End")


def pars_args():
    arguments = argparse.ArgumentParser()
    arguments.add_argument(
        '-s', '--seed', type=int, default=1,
        help="Seed for the random number generator"
    )
    arguments.add_argument(
        '-dp', '--DataPath', type=str,
        help="Complete path to dataset",
        required=True
    )
    arguments.add_argument(
        '-sp', '--SavePath', type=str,
        help="Complete path to directory where to store computation",
        required=True
    )
    arguments.add_argument(
        '-bs', "--BatchSize", type=int, default=64,
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
    arguments.add_argument(
        '-logp', "--LoggingPath", type=str, default="$HOME/Desktop/log_gan",
        help="Complete path to directory where to load info"
    )
    return arguments.parse_args()


if __name__ == "__main__":
    main(pars_args())
