import argparse
import numpy as np
import theano.tensor as T
import theano
import lasagne
from lasagne.layers import Conv2DLayer, DenseLayer, batch_norm, InputLayer, ReshapeLayer, Deconv2DLayer
from lasagne.nonlinearities import sigmoid
from lasagne.objectives import binary_crossentropy
import os
import PIL.Image as Image
import _pickle as pickle
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


def generator(net):
    """
    Function that build the generator network
    :param input_var: Input variable that goes in Lasagne.layers.InputLayer
    """
    # net = InputLayer(shape=(None, 100), input_var=input_var)
    # net = batch_norm(DenseLayer(net, 1024))
    # Project
    net = batch_norm(DenseLayer(net, 1024*4*4))
    # Reshape
    net = ReshapeLayer(net, ([0], 1024, 4, 4))

    net = batch_norm(Deconv2DLayer(net, 128, 4, stride=4))
    net = batch_norm(Deconv2DLayer(net, 64, 2, stride=2))
    net = Deconv2DLayer(net, 3, 2, stride=2, nonlinearity=sigmoid)

    print("Generator output shape: ", net.output_shape)
    return net


def discriminator(input_var=None):
    """
    Function that build the discriminator
    :param input_var: Input variable that goes in Lasagne.layers.InputLayer
    """
    lrelu = lasagne.nonlinearities.LeakyRectify()
    net = InputLayer((None, 3, 64, 64), input_var=input_var)
    # 128 units of 32 x 32
    net = batch_norm(Conv2DLayer(net, 128, 2, stride=2))
    # 128 uints of 32 x 32
    net = batch_norm(Conv2DLayer(net, 128, 3, pad=1))
    # 256 units of 16 x 16
    net = batch_norm(Conv2DLayer(net, 256, 2, stride=2))
    # 256 units of 16 x 16
    net = batch_norm(Conv2DLayer(net, 256, 3, pad=1))
    # 512 units of 8 x 8
    net = batch_norm(Conv2DLayer(net, 512, 2, stride=2))
    # 512 units of 8 x 8
    net = batch_norm(Conv2DLayer(net, 512, 3, pad=1))
    # 1024 units of 4 x 4
    net = batch_norm(Conv2DLayer(net, 1024, 2, stride=2))
    # Fully connected layers
    net = batch_norm(DenseLayer(net, 1024, nonlinearity=lrelu))
    net = DenseLayer(net, 1, nonlinearity=sigmoid)
    # net = batch_norm(Conv2DLayer(net, 64, 5, stride=2, pad=2, nonlinearity=lrelu))
    # net = batch_norm(Conv2DLayer(net, 64, 5, stride=2, pad=2, nonlinearity=lrelu))
    # net = batch_norm(DenseLayer(net, 1024, nonlinearity=lrelu))
    # net = DenseLayer(net, 1, nonlinearity=sigmoid)
    print("Discriminator output shape : ", net.output_shape)

    return net


def minibatch_iterator(x, y, batch_size):
    """
    Iterator on a minibatch.

    :param x: Input data to fetch
    :param y: Target data to fetch
    :param batch_size: Size of a batch
    :return: Two arrays containing the input and target
    """
    load_data = data_utils.load_data
    assert len(x) == len(y)
    i = None
    for i in range(0, len(x) - batch_size + 1, batch_size):
        batch = slice(i, i + batch_size)
        yield load_data(x[batch], (64, 64)), load_data(y[batch], (64, 64))
    # Make sure that all the dataset is passed
    # even if it is less then a full batch_size
    if i is None:
        i = 0
    # Fetch the last data from the dataset
    if i < len(x):
        batch = slice(i, len(x))
        yield load_data(x[batch], (64, 64)), load_data(y[batch], (64, 64))


def input_iterator(x, batch_size):
    """
    Iterator on an input data.

    :param x: Input data to fetch
    :param batch_size: Size of a batch
    :return: Array of data
    """
    load_data = data_utils.load_data

    i = None
    for i in range(0, len(x) - batch_size + 1, batch_size):
        batch = slice(i, i + batch_size)
        yield load_data(x[batch], (64, 64))
    # Make sure that all the dataset is passed
    # even if it is less then a full batch_size
    if i is None:
        i = 0
    # Fetch the last data from the dataset
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
            self.data_path = data_path
        else:
            raise ValueError("{0} is not a path".format(data_path))
        # Verify directory of save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        self.save_path = save_path

        # Load parameters if load_file is given
        if load_file is not None:
            # self.load_params(load_file)
            self.load_params = load_file

    def generate_images(self, noise):
        print("Building the generator")
        gen = generator(noise)
        # with np.load(self.load_params + '/gan_gen.npz') as f:
        #     gen_param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(gen, gen_param_values)
        #
        # print("Building the function")
        # gen_fn = theano.function(
        #     [noise],
        #     lasagne.layers.get_output(gen, deterministic=True)
        # )
        # print("Generate Image")
        # samples = gen_fn(lasagne.utils.floatX(np.random.rand(10, 100)))
        # try:
        #     import matplotlib.pyplot as plt
        # except ImportError:
        #     pass
        # else:
        #     plt.imsave('samples.png',
        #     (samples.reshape(10, 3, 64, 64)))

    def train(self, epochs, batch_size=128, learning_rate=2e-4):
        # list_of_image = glob.glob(self.data_path + "/train2014" + "/input_*.jpg")
        # list_of_targets = glob.glob(self.data_path + "/train2014" + "/target_*.jpg")
        # list_of_image = glob.glob(self.data_path + "/input_*.jpg")
        # list_of_targets = glob.glob(self.data_path + "/target_*.jpg")
        print("Loading Data to ram")
        dic = pickle.load(open(self.data_path + '/data.pkl', 'rb'))
        prefixes = ['/input_', '/img_']
        data = data_utils.load_data_to_ram(dic=dic, prefixes=prefixes, data_path=self.data_path)

        # assert len(list_of_image) is not 0
        # assert len(list_of_image) == len(list_of_targets)

        # n_batch = len(list_of_targets) // batch_size

        # noise = T.matrix('noise')
        x = T.tensor4('x')
        y = T.tensor4('image')
        # target = T.tensor4('target')

        x = x.reshape((batch_size, 3, 64, 64))
        y = y.reshape((batch_size, 3, 64, 64))

        print("Building the model")
        noise = image_encoder(x)
        gen = generator(noise)

        dis = discriminator(y)

        # Theano Function that output the real and fake value
        # from the discriminator and the generator
        real = lasagne.layers.get_output(dis, y)
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

        print("Compiling function")
        # Theano function that performs a training on a minibatch
        train_fn = theano.function(
            [x, y],
            [(real > .5).mean(),
             (fake < .5).mean()],
            updates=updates
        )

        # Theano function that creates data
        gen_fn = theano.function(
            [x],
            lasagne.layers.get_output(gen, deterministic=True)
        )

        print("Begining training")
        # Training loop
        for e in range(epochs):
            b = 0
            err = 0
            tic = time.time()
            for batch in data_utils.minibatch_iterator(x=data[0], y=data[1], batch_size=batch_size):
                inputs, target = batch
                inputs = inputs.astype(np.float32)
                target = target.astype(np.float32)
                err += np.array(train_fn(target, inputs))
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

        output = gen_fn(data[0].astype(np.float32))

        # for i in output:
        #     img = np.asarray(i*255, dtype=np.uint8)
        #     img = Image.fromarray(img.reshape(64, 64, 3))
        #     img.show()


        print("End of training")


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
    g = GAN(
        data_path=args.DataPath,
        save_path=args.SavePath,
        load_file=args.LoadPath
    )

    if args.train is not 0:
        g.train(
            epochs=args.epochs,
            batch_size=args.BatchSize,
            learning_rate=args.LearningRate
        )
    else:
        noise = T.matrix('noise')
        g.generate_images(noise)

    return

if __name__ == '__main__':
    main(pars_args())
