import os
import argparse
import logging
import time
import numpy as np
import PIL.Image as Image
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import Conv2DLayer, DenseLayer, batch_norm, InputLayer, ReshapeLayer, Deconv2DLayer
from lasagne.nonlinearities import sigmoid, rectify, leaky_rectify, tanh

import _pickle as pickle
# import six.moves.cPickle as pickle

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
        for j in range(input.shape[1]):
            input[i, j,
            center[0] - 16: center[0] + 16,
            center[1] - 16: center[1] + 16] = np.random.random((32, 32))

    return input


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

    logging.info("Image encoder output shape: {}".format(net.output_shape))
    return net


def image_decoder(net=None, input_var=None):

    # Output size of deconvolution formula:
    # o = s(i - 1) + a + k - 2p
    # Where o is the output size, i the input, p
    # the padding, k the kernel size and s the stride

    tanh = lasagne.nonlinearities.tanh
    if net is None:
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

    logging.info("Image decoder output shape: {}".format(net.output_shape))
    return net


def generator(net, input_var=None):
    """
    Function that build the generator network
    :param input_var: Input variable that goes in Lasagne.layers.InputLayer
    """
    if net is None:
        net = InputLayer(shape=(None, 100), input_var=input_var)
    # net = batch_norm(DenseLayer(net, 1024))
    # Project
    net = batch_norm(DenseLayer(net, 1024*4*4))
    # Reshape
    net = ReshapeLayer(net, ([0], 1024, 4, 4))
    # 512 units of 8 x 8
    net = batch_norm(Deconv2DLayer(net, 512, 2, stride=2, nonlinearity=rectify))
    # net = batch_norm(Conv2DLayer(net, 512, 3, pad=1, nonlinearity=rectify))
    # 256 units of 16 x 16
    net = batch_norm(Deconv2DLayer(net, 256, 2, stride=2, nonlinearity=rectify))
    # net = batch_norm(Conv2DLayer(net, 256, 3, pad=1, nonlinearity=rectify))
    # 128 units of 16 x 16
    net = batch_norm(Conv2DLayer(net, 128, 3, pad=1, nonlinearity=rectify))
    # 64 units of 32 x 32
    net = Deconv2DLayer(net, 3, 2, stride=2, nonlinearity=tanh)

    logging.info("Generator output shape: {}".format(net.output_shape))
    return net


def discriminator(input_var=None):
    """
    Function that build the discriminator
    :param input_var: Input variable that goes in Lasagne.layers.InputLayer
    """

    # Output size of convolution formula:
    # o = (i + 2p - k) / s + 1
    # Where o is the output size, i the input, p
    # the padding, k the kernel size and s the stride

    lrelu = leaky_rectify
    net = InputLayer((None, 3, 32, 32), input_var=input_var)
    # 128 units of 16 x 16
    net = batch_norm(Conv2DLayer(net, 128, 2, stride=2, nonlinearity=lrelu))
    # 256 units of 8 x 8
    net = batch_norm(Conv2DLayer(net, 256, 2, stride=2, nonlinearity=lrelu))
    # 512 units of 4 x 4
    # net = batch_norm(Conv2DLayer(net, 512, 2, stride=2))
    # 512 units of 8 x 8
    # net = batch_norm(Conv2DLayer(net, 512, 3, pad=1))
    # 1024 units of 4 x 4
    # net = batch_norm(Conv2DLayer(net, 1024, 2, stride=2))
    # Fully connected layers
    net = batch_norm(DenseLayer(net, 256, nonlinearity=lrelu))
    net = DenseLayer(net, 1, nonlinearity=sigmoid)

    logging.info("Discriminator output shape : {}".format( net.output_shape))

    return net


def set_params(net, param_file):
    with np.load(param_file) as f:
        gen_param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net, gen_param_values)
    return net


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

    def generate_images(self, indexes):
        dic = pickle.load(open(self.data_path + '/data.pkl', 'rb'))
        pre = self.data_path + '/input_'

        logging.info("Building the generator")
        x = T.tensor4('x')
        #x = x.reshape((128, 3, 64, 64))
        enc = image_encoder(x)
        enc = set_params(enc, self.load_params + '/GAN2_enc.npz')

        gen = generator(net=enc)

        with np.load(self.load_params + '/GAN2_gen.npz') as f:
            gen_param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(gen, gen_param_values)

        logging.info("Building the function")
        gen_fn = theano.function(
            [x],
            lasagne.layers.get_output(gen, deterministic=True)
        )
        logging.info("Generate Image")
        list_of_inputs = [pre + dic.get(key.astype(np.int32)) + '.jpg' for key in indexes]
        data = data_utils.load_data(list_of_images=list_of_inputs, size=(64, 64))
        data = add_noize(data)
        samples = gen_fn(lasagne.utils.floatX(data))

        for img in samples:
            img *= 255
            img = img.reshape(32, 32, 3)
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img.show()

        logging.info("End of generation")

    def train(self, epochs, batch_size=128, learning_rate=2e-4):
        dic = pickle.load(open(self.data_path + '/data.pkl', 'rb'))
        prefixes = ['/input_', '/target_']

        x = T.tensor4('x')
        y = T.tensor4('y')

        x.reshape((batch_size, 3, 64, 64))
        y.reshape((batch_size, 3, 32, 32))

        enc = image_encoder(x)
        gen = generator(net=enc)
        dis = discriminator()

        real = lasagne.layers.get_output(dis, y)
        fake = lasagne.layers.get_output(dis, lasagne.layers.get_output(gen))

        gen_cost_dis = lasagne.objectives.binary_crossentropy(fake, 1).mean()
        dis_cost_gen = lasagne.objectives.binary_crossentropy(fake, 0).mean()
        dis_cost_real = lasagne.objectives.binary_crossentropy(real, 1).mean()

        # Encoder cost with MSE
        test_input = lasagne.layers.InputLayer((None, 3, 32, 32), y)
        test = lasagne.layers.Pool2DLayer(test_input, (4, 4), mode='average_exc_pad', ignore_border=True)
        test_flatten = lasagne.layers.FlattenLayer(test, 2)

        test_g = lasagne.layers.Pool2DLayer(gen, (4, 4), mode='average_exc_pad', ignore_border=True)
        test_g_flatten = lasagne.layers.FlattenLayer(test_g, 2)

        test_output, test_g_output = lasagne.layers.get_output([test_flatten, test_g_flatten])
        enc_cost = lasagne.objectives.squared_error(test_g_output, test_output).mean()

        dis_cost = dis_cost_real + dis_cost_gen
        gen_cost = gen_cost_dis + enc_cost / 100

        # Create updaters
        shrd_lr = theano.shared(lasagne.utils.floatX(learning_rate))
        dis_params = lasagne.layers.get_all_params(dis, trainable=True)

        updates_dis = lasagne.updates.adam(
            dis_cost, dis_params, learning_rate=shrd_lr, beta1=0.8, beta2=0.9
        )

        enc_gen_params = lasagne.layers.get_all_params([enc, gen], trainable=True)
        updates_gen = lasagne.updates.adam(
            gen_cost, enc_gen_params, learning_rate=shrd_lr, beta1=0.8, beta2=0.9
        )

        eshrd_lr = theano.shared(lasagne.utils.floatX(0.002))
        enc_dec_params = lasagne.layers.get_all_params(enc, trainable=True)
        updates_enc = lasagne.updates.adam(
            enc_cost, enc_dec_params, learning_rate=eshrd_lr, beta1=0.8, beta2=0.9
        )

        logging.info("Compiling function")

        train_gen_fn = theano.function(
            [x, y],
            gen_cost_dis,
            updates=updates_gen
        )
        train_dis_fn = theano.function(
            [x, y],
            dis_cost_real,
            updates=updates_dis
        )
        train_enc_fn = theano.function(
            [x, y],
            enc_cost,
            updates=updates_enc
        )

        g_costs = []
        d_costs = []
        e_costs = []
        upd = 0
        logging.info("Beginning Training")
        # Training loop
        for e in range(epochs):
            np.random.seed(34)
            tic = time.time()
            for data in data_utils.load_data_to_ram(
                length=10 * batch_size,
                dic=dic,
                prefixes=prefixes,
                data_path=self.data_path,
                size=[(64, 64), (32, 32)]
            ):
                for batch in data_utils.minibatch_iterator(x=data[0], y=data[1], batch_size=batch_size):
                    inputs, targets = batch
                    inputs = add_noize(inputs)
                    inputs = inputs.astype(np.float32)
                    targets = targets.astype(np.float32)

                    cost_g = train_gen_fn(inputs, targets)
                    cost_d = train_dis_fn(inputs, targets)
                    cost_e = train_enc_fn(inputs, targets)

                    g_costs.append(cost_g)
                    d_costs.append(cost_d)
                    e_costs.append(cost_e)

                    logging.info("Epoch {} of {}, elapsed time: {:.3f} minutes".format(
                        e + 1, epochs, (time.time() - tic)/60
                    ))

                    logging.info("Generator error: {}, Discriminator error: {}, Encoder error: {}".format(
                        cost_g, cost_d, cost_e
                    ))

                    shrd_lr.set_value(lasagne.utils.floatX(shrd_lr.get_value() * 0.95))
                    # upd += 1

            np.savez(self.save_path + '/GAN2_gen.npz', *lasagne.layers.get_all_param_values(gen))
            np.savez(self.save_path + '/GAN2_disc.npz', *lasagne.layers.get_all_param_values(dis))
            np.savez(self.save_path + '/GAN2_enc.npz', *lasagne.layers.get_all_param_values(enc))

        logging.info("End of training")


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
        '-logp', "--LoggingPath", type=str, default="$HOME/Desktop/log_gan",
        help="Complete path to directory where to load info"
    )
    return arguments.parse_args()


def main(args):
    args.LoggingPath = os.path.expandvars(args.LoggingPath)
    post = 'log_' + time.strftime('%m_%d_%Y_%H_%M_%S') + '_gan2.log'

    if not os.path.isdir(args.LoggingPath):
        os.mkdir(args.LoggingPath)

    g = GAN(
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
        indexes = np.arange(0, 10)
        g.generate_images(indexes=indexes)
    else:
        logname = '/Generation_' + post
        create_logging(args.LoggingPath, logname)
        #indexes = np.random.randint(0, 80000, 10)
        indexes = np.arange(0, 10)
        g.generate_images(indexes=indexes)

    return

if __name__ == '__main__':
    main(pars_args())