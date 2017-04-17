import theano
import theano.tensor as T
import numpy as np
import argparse
import logging
import os
import time
import PIL.Image as Image

import _pickle as pickle
# import six.moves.cPickle as pickle

import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, batch_norm, DenseLayer, Deconv2DLayer, ReshapeLayer
from lasagne.nonlinearities import sigmoid

try:
    from models import data_utils
except ImportError:
    from utils import data_utils


def image_encoder(input_var=None):
    """
    Function that build the encoder network
    :param input_var: 
    :return: The encoder network
    """

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

    logging.info("Image encoder output shape: {}".format( net.output_shape))
    return net


def image_decoder(net=None, input_var=None):
    """
    Function that build the generator network
    :param net: Network that creates the encoded image
    :param input_var: Input variable that goes in Lasagne.layers.InputLayer
    :return: Tne decoder network
    """

    # Output size of deconvolution formula:
    # o = s(i - 1) + a + k - 2p
    # Where o is the output size, i the input, p
    # the padding, k the kernel size and s the stride

    if net is None:
        net = InputLayer(shape=(None, 100), input_var=input_var)

    # net = InputLayer(shape=(None, 100), input_var=input_var)
    # net = batch_norm(DenseLayer(net, 1024))
    # Project
    net = batch_norm(DenseLayer(net, 1024 * 4 * 4))
    # Reshape
    net = ReshapeLayer(net, ([0], 1024, 4, 4))
    # 512 units of 8 x 8
    net = batch_norm(Deconv2DLayer(net, 512, 2, stride=2))
    # net = batch_norm(Conv2DLayer(net, 512, 3, pad=1))
    # 256 units of 16 x 16
    net = batch_norm(Deconv2DLayer(net, 256, 9))
    # net = batch_norm(Conv2DLayer(net, 256, 3, pad=1))
    # 128 units of 16 x 16
    net = batch_norm(Conv2DLayer(net, 128, 3, pad=1))
    # 64 units of 32 x 32
    net = batch_norm(Deconv2DLayer(net, 64, 2, stride=2))
    # 3 units of 64 x 64
    net = Deconv2DLayer(net, 3, 2, stride=2, nonlinearity=sigmoid)

    logging.info("Generator output shape: {}".format(net.output_shape))
    return net


class VAE(object):
    """
    
    """
    def __init__(self, save_path, data_path, load_file=None, seed=0):
        """
        
        :param save_path: 
        :param data_path: 
        :param load_path: 
        """
        self.save_path = save_path
        self.data_path = data_path
        self.load_file = load_file

        np.random.seed(seed)

    def train(self, epochs=10, batch_size=128, learning_rate=0.0001):
        """
        
        :param epochs: 
        :param batch_size: 
        :param learning_rate: 
        """

        dic = pickle.load(open(self.data_path + '/data.pkl', 'rb'))
        prefixes = ['/img_', '/img_']
        x = T.tensor4('x')

        x = x.reshape((batch_size, 3, 64, 64))

        # Building the network
        logging.info("Building Network")
        encoder = image_encoder(x)
        decoder = image_decoder(encoder)

        output = lasagne.layers.get_output(decoder, x)
        params = lasagne.layers.get_all_params(decoder, trainable=True)
        loss = lasagne.objectives.squared_error(output, x).mean()
        lb = theano.shared(lasagne.utils.floatX(learning_rate))
        updates = lasagne.updates.adam(
            loss, params, learning_rate=lb
        )

        # Creating trainable funciton
        logging.info("Compiling function")
        train_fn = theano.function(
            [x],
            [loss],
            updates=updates
        )

        # Theano function that creates data
        gen_fn = theano.function(
            [x],
            lasagne.layers.get_output(decoder, deterministic=True)
        )

        # Beginning training
        logging.info("Beginning training")
        for e in range(epochs):
            b = 0
            err = 0
            tic = time.time()
            for data in data_utils.load_data_to_ram(
                length=10*batch_size,
                dic=dic,
                prefixes=prefixes,
                data_path=self.data_path
            ):
                for batch in data_utils.minibatch_iterator(
                    x=data[0],
                    y=data[1],
                    batch_size=batch_size
                ):
                    inputs, targets = batch
                    inputs = inputs.astype(np.float32)
                    targets = targets.astype(np.float32)
                    err += np.array(train_fn(inputs))
                    b += 1

            logging.info("Epoch {} of {}, elapsed time: {:.3f} minutes".format(
                e + 1, epochs, (time.time() - tic) / 60
            ))

            logging.info("Loss {}".format(err / b))

            np.savez(self.save_path + '/encoder.npz', *lasagne.layers.get_all_param_values(encoder))
            np.savez(self.save_path + '/decoder.npz', *lasagne.layers.get_all_param_values(decoder))

        logging.info("End of training")

        out = gen_fn(inputs)

        for img in out:
            img = img * 255
            img = img.astype(np.uint8)
            img = img.reshape((64, 64, 3))
            img = Image.fromarray(img)
            img.show()


def create_logging(LoggingPath, logname):
    logpath = LoggingPath + logname
    logging.basicConfig(filename=logpath,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')


def pars_args():
    arguments = argparse.ArgumentParser()
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
        '-logp', "--LoggingPath", type=str, default="$HOME/Desktop/log_VAE",
        help="Complete path to directory where to load info"
    )
    return arguments.parse_args()


def main(args):
    args.LoggingPath = os.path.expandvars(args.LoggingPath)
    post = 'log_' + time.strftime('%m_%d_%Y_%H_%M_%S') + '_auto_encoder.log'

    if not os.path.isdir(args.LoggingPath):
        os.mkdir(args.LoggingPath)

    g = VAE(
        data_path=args.DataPath,
        save_path=args.SavePath,
        load_file=args.LoadPath
    )

    if args.train is not 0:
        logname = '/Training_' + post
        create_logging(args.LoggingPath, logname)
        g.train(
            epochs=args.epochs,
            batch_size=args.BatchSize,
            learning_rate=args.LearningRate
        )
    else:
        logname = '/Generation_' + post
        create_logging(args.LoggingPath, logname)
        #indexes = np.random.randint(0, 80000, 10)
        indexes = np.arange(0, 10)
        g.generate_images(indexes=indexes)

    return

if __name__ == '__main__':
    main(pars_args())
