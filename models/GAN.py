import argparse
import numpy as np
import theano.tensor as T
import theano
import lasagne
from lasagne.layers import Deconv2DLayer, Conv2DLayer, DenseLayer, batch_norm, InputLayer, ReshapeLayer, DilatedConv2DLayer
from lasagne.nonlinearities import sigmoid
from lasagne.objectives import binary_crossentropy
import os
import glob
import time

try:
    from models.utils import data_utils
except ImportError:
    from utils import data_utils

"""
Generative Adversarial Network (GAN)

The GAN network is made of 2 models that compete
against each other. The first network is a Generative
network and the second is a discriminative network.
The generative network have the goals to generate
data that is realistic enough to fools the discriminative
network into thinking it is real data (data that comes from
the same statistical distribution then the real data it receives
from samples).
"""


def generator(input_var=None):
    """
    Function that build the generator network
    :param input_var: Input variable that goes in Lasagne.layers.InputLayer
    """
    net = InputLayer(shape=(None, 100), input_var=input_var)
    net = batch_norm(DenseLayer(net, 1024))
    # Project
    net = batch_norm(DenseLayer(net, 3*64*64))
    # Reshape
    net = ReshapeLayer(net, ([0], 3, 64, 64))

    net = batch_norm(DilatedConv2DLayer(net, 64, 1, dilation=1))
    net = DilatedConv2DLayer(net, 3, 1, dilation=1, nonlinearity=sigmoid)

    print(net.output_shape)
    return net


def discriminator(input_var=None):
    """
    Function that build the discriminator
    :param input_var: Input variable that goes in Lasagne.layers.InputLayer
    """
    lrelu = lasagne.nonlinearities.LeakyRectify()
    net = InputLayer((None, 3, 64, 64), input_var=input_var)
    net = batch_norm(Conv2DLayer(net, 64, 5, stride=2, pad=2, nonlinearity=lrelu))
    net = batch_norm(Conv2DLayer(net, 64, 5, stride=2, pad=2, nonlinearity=lrelu))
    net = batch_norm(DenseLayer(net, 1024, nonlinearity=lrelu))
    net = DenseLayer(net, 1, nonlinearity=sigmoid)
    print(net.output_shape)

    return net


def minibatch_iterator(x, y, batch_size):
    load_data = data_utils.load_data
    assert len(x) == len(y)
    i = None
    for i in range(0, len(x) - batch_size + 1, batch_size):
        batch = slice(i, i + batch_size)
        yield load_data(x[batch], (64, 64)), load_data(y[batch], (64, 64))
    # Assure that if the dataset is
    if i is None:
        print("None")
        i = 0

    if i < len(x):
        batch = slice(i, len(x))
        yield load_data(x[batch], (64, 64)), load_data(y[batch], (64, 64))

def input_iterator(x, batch_size):
    load_data = data_utils.load_data

    i = None
    for i in range(0, len(x) - batch_size + 1, batch_size):
        batch = slice(i, i + batch_size)
        yield load_data(x[batch], (64, 64))
    # Assure that if the dataset is
    if i is None:
        print("None")
        i = 0

    if i < len(x):
        batch = slice(i, len(x))
        yield load_data(x[batch], (64, 64))

class GAN(object):
    """
    GAN class, creates a GAN to use with MSCOCO dataset.
    """
    def __init__(self, data_path="../data/input", save_path="./tmp", load_file=None):
        """
        Constructor of the GAN class

        :param data_path: Path to the dataset
        :param save_path: Path to directory where to save important data
        :param load_file: Path to a file with pretrained parameters
        """
        # Verify directory of data_set
        if data_utils.verify_dataset(data_path):
            self.data_path=data_path
        else:
            raise ValueError("{0} is not a path".format(data_path))
        # Verify directory of save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        self.save_path=save_path

        # Load parameters if load_file is given
        if load_file is not None:
            self.load_params(load_file)

    def generate_images(self, noise):
        print("Generate Image")

    def train(self, epochs, batch_size=128, learning_rate=2e-4):
        # list_of_image = glob.glob(self.data_path + "/train2014" + "/input_*.jpg")
        # list_of_targets = glob.glob(self.data_path + "/train2014" + "/target_*.jpg")
        list_of_image = glob.glob(self.data_path + "/*.jpg")
        # list_of_targets = glob.glob(self.data_path + "/target_*.jpg")

        assert len(list_of_image) is not 0
        # assert len(list_of_image) == len(list_of_targets)

        # n_batch = len(list_of_targets) // batch_size

        noise = T.matrix('noise')
        x = T.tensor4('x')
        # target = T.tensor4('target')

        x = x.reshape((batch_size, 3, 64, 64))

        print("Building the model")
        gen = generator(noise)
        dis = discriminator(x)

        # Theano Function that output the real and fake value
        # from the discriminator and the generator
        real = lasagne.layers.get_output(dis, x)
        fake = lasagne.layers.get_output(dis, lasagne.layers.get_output(gen))

        # Create loss expressions
        gen_loss = binary_crossentropy(fake, 1).mean()
        dis_loss = (binary_crossentropy(real, 1) + binary_crossentropy(fake, 0)).mean()

        # Create update expressions
        gen_params = lasagne.layers.get_all_params(gen, trainable=True)
        dis_params = lasagne.layers.get_all_params(dis, trainable=True)
        eta = theano.shared(lasagne.utils.floatX(learning_rate))
        updates = lasagne.updates.adam(
            gen_loss, gen_params, learning_rate=eta, beta1=0.5
        )
        updates.update(
            lasagne.updates.adam(
                dis_loss, dis_params, learning_rate=eta, beta1=0.5
            )
        )

        # Theano function that performs a training on a minibatch
        train_fn = theano.function(
            [noise, x],
            [(real > .5).mean(),
             (fake < .5).mean()],
            updates=updates
        )

        # Theano function that creates data
        # gen_fn = theano.function(
        #     [noise],
        #     lasagne.layers.get_output(gen, deterministic=True)
        # )

        print("Begining training")
        # Training loop
        for e in range(epochs):
            b = 0
            err = 0
            tic = time.time()
            for batch in input_iterator(list_of_image, 128):
                inputs = batch # , target = batch
                inputs = inputs.astype(np.float32)
                # target = target.astype(np.float32)
                noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))
                err += np.array(train_fn(noise, inputs))
                b += 1

            print("Epoch {} of {}, elapsed time: {:.3f} seconds".format(
                e, epochs, time.time() - tic
            ))

            print("Loss {}".format(err / b))

            if e >= epochs // 2:
                progress = float(e) / epochs
                eta.set_value(lasagne.utils.floatX(learning_rate * 2 * (1 - progress)))

            np.savez(self.save_path + '/GAN_gen.npz', *lasagne.layers.get_all_param_values(gen))
            np.savez(self.save_path + '/GAN_disc.npz', *lasagne.layers.get_all_param_values(dis))


def pars_args():
    arguments = argparse.ArgumentParser();
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
        '-t', '--train', type=bool, default=True,
        help="Define if the GAN must be trained"
    )
    arguments.add_argument(
        '-lr', '--LearningRate', type=float, default=2e-4,
        help="Defines the learning rate of the GAN network"
    )
    return arguments.parse_args()

def main(args):
    g = GAN(args.DataPath, args.SavePath)

    if args.train:
        g.train(
            epochs=args.epochs,
            batch_size=args.BatchSize,
            learning_rate=args.LearningRate
        )


if __name__ == '__main__':
    main(pars_args())
