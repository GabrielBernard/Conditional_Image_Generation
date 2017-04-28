"""
Script that creates a DCGAN network.

This script is made to be use on the
MSCOCO dataset downsampled to 64x64 pixels.

Author: Gabriel Bernard
Updated on: 2017-04-28
"""

import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, Deconv2DLayer, BatchNormLayer, ReshapeLayer
from lasagne.nonlinearities import LeakyRectify, rectify, sigmoid, tanh
import logging
import six.moves.cPickle as pickle
import argparse
import PIL.Image as Image

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


# Global argument that controls the logging function
log_b = False


def set_log(b):
    """
    Sets the log_b global variable.

    :param b: boolean valuethat decides whether to
        log in a file or to print in the console.
    """

    global log_b
    log_b = b


def log_fn(s):
    """
    Log function

    :param s: String to print
    """

    if log_b:
        # Log in file
        logging.info(s)
    else:
        # Print in console
        print(s)


def add_noize(input):
    """
    Function that adds noize in the middle of the
    input images.

    :param input: Numpy array of shape (batch_size, 3, 64, 64)
    :return input with noise in the middle
    """

    center = (
        int(np.floor(input.shape[2] / 2.)),
        int(np.floor(input.shape[3] / 2.))
    )

    # input[:,
    #       center[0] - 16: center[0] + 16,
    #       center[1] - 16: center[1] + 16, :
    # ] = np.random.random((input.shape[0], 3, 32, 32) + 1) / 100
    # Loops that fills the input with random noise in the middle
    for i in range(input.shape[0]):
        input[i, :,
        center[0] - 16: center[0] + 16,
        center[1] - 16: center[1] + 16] = (np.random.random((3, 32, 32)) + 1) / 100

    return input


def image_encoder(input_var=None):
    """
    Function that builds an image encoder.

    :param input_var: Input variable that goes in Lasagne.layers.InputLayer
    """

    # Output size of convolution formula:
    # o = (i + 2p - k) / s + 1
    # Where o is the output size, i the input, p
    # the padding, k the kernel size and s the stride

    net = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
    # 128 units of 32 x 32
    net = BatchNormLayer(Conv2DLayer(net, 128, 2, stride=2))
    # 256 units of 16 x 16
    net = BatchNormLayer(Conv2DLayer(net, 256, 2, stride=2))
    # 512 units of 8 x 8
    net = BatchNormLayer(Conv2DLayer(net, 512, 2, stride=2))
    # 1024 units of 4 x 4
    net = BatchNormLayer(Conv2DLayer(net, 1024, 2, stride=2))
    # Fully connected layer
    net = DenseLayer(net, 100, nonlinearity=tanh)

    log_fn("Image encoder output shape: {}".format(net.output_shape))
    return net


def image_decoder(net=None, input_var=None):
    """
    Function that build an image decoder.

    :param net: If a net is given, it will be used as input layer to build
        the rest of the network
    :param input_var: Input variable that goes in Lasagne.layers.InputLayer
    """

    # Output size of deconvolution formula:
    # o = s(i - 1) + a + k - 2p
    # Where o is the output size, i the input, p
    # the padding, k the kernel size and s the stride

    if net is None:
        net = InputLayer(shape=(None, 100), input_var=input_var)
    # Project
    net = BatchNormLayer(DenseLayer(net, 1024 * 4 * 4, nonlinearity=tanh))
    # Reshape
    net = ReshapeLayer(net, ([0], 1024, 4, 4))
    # 512 units of 8 x 8
    net = BatchNormLayer(Deconv2DLayer(net, 512, 2, stride=2))
    # 256 units of 16 x 16
    net = BatchNormLayer(Deconv2DLayer(net, 256, 9))
    # 128 units of 32 x 32
    net = BatchNormLayer(Deconv2DLayer(net, 128, 2, stride=2))
    # 3 units of 64 x 64 (rgb image)
    net = Deconv2DLayer(net, 3, 2, stride=2)

    log_fn("Image decoder output shape: {}".format(net.output_shape))
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

    lrelu = LeakyRectify(0.2)
    net = InputLayer((None, 3, 64, 64), input_var=input_var)
    # 128 units of 16 x 16
    net = Conv2DLayer(net, 128, 2, stride=2, nonlinearity=lrelu)
    # 256 units of 8 x 8
    net = BatchNormLayer(Conv2DLayer(net, 256, 2, stride=2, nonlinearity=lrelu))
    # 512 units of 4 x 4
    net = BatchNormLayer(Conv2DLayer(net, 512, 2, stride=2, nonlinearity=lrelu))
    # 512 units of 8 x 8
    # net = batch_norm(Conv2DLayer(net, 512, 3, pad=1, nonlinearity=lrelu))
    # 1024 units of 4 x 4
    net = BatchNormLayer(Conv2DLayer(net, 1024, 2, stride=2, nonlinearity=lrelu))
    # Fully connected layers
    net = DenseLayer(net, 1024, nonlinearity=lrelu)
    # Layer that computes the probability of the image being real
    net = DenseLayer(net, 1, nonlinearity=sigmoid)

    log_fn("Discriminator output shape : {}".format( net.output_shape))

    return net


def generator(input_var=None):
    """
    Function that build the generator network

    :param input_var: Input variable that goes in Lasagne.layers.InputLayer
    """

    net = InputLayer(shape=(None, 100), input_var=input_var)
    # Projection
    net = BatchNormLayer(DenseLayer(net, 1024*4*4))
    # Reshape
    net = BatchNormLayer(ReshapeLayer(net, ([0], 1024, 4, 4)))
    # 512 units of 8 x 8
    net = BatchNormLayer(Deconv2DLayer(net, 512, 2, stride=2, nonlinearity=rectify))
    # 256 units of 16 x 16
    net = BatchNormLayer(Deconv2DLayer(net, 256, 2, stride=2, nonlinearity=rectify))
    # 128 units of 16 x 16
    net = BatchNormLayer(Conv2DLayer(net, 128, 3, pad=1, nonlinearity=rectify))
    # 64 units of 32 x 32
    net = Deconv2DLayer(net, 64, 2, stride=2, nonlinearity=tanh)
    # 3 units of 64 x 64 (aka generated image)
    net = Deconv2DLayer(net, 3, 2, stride=2, nonlinearity=tanh)

    log_fn("Generator output shape: {}".format(net.output_shape))
    return net


def create_logging(LoggingPath, logname):
    """
    Function that creates the logging path and file.

    :param LoggingPath: Path to where the log file must be created
    :param logname: name of the file to be created
    """

    # Checking the path for environment variables
    LoggingPath = os.path.expandvars(LoggingPath)

    # If path is does not exist, creates it
    if not os.path.isdir(LoggingPath):
        os.mkdir(LoggingPath)

    # Creates the log file
    logpath = LoggingPath + logname
    logging.basicConfig(
        filename=logpath,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    )


def training_encoder(batch_size, dic, prefixes, data_path, train_encode, save_path, encoder, decoder):

    num_iter = np.round((len(dic) - 1) / batch_size).astype(np.int)

    log_fn("Beginning Training of encoder")
    for epoch in range(1):
        # train_both = True
        train_gen = True
        ind = 0
        it = 1
        loss = 0.
        for data in data_utils.load_data_to_ram(
                length=10 * batch_size,
                dic=dic,
                prefixes=prefixes,
                data_path=data_path,
                size=[(64, 64), (32, 32)]
        ):
            for batch in data_utils.minibatch_iterator(x=data[0], y=data[1], batch_size=batch_size):
                inputs, targets = batch
                inputs = add_noize(inputs)
                inputs = inputs.astype(np.float32)
                targets = None

                loss += train_encode(inputs)
                if (it % 1) == 0:
                    log_fn("Epoch: {} of {}, it {} of {}, Loss : {}".format(
                        epoch + 1, 1,
                        it, num_iter, loss / it)
                    )

                it += 1

        np.savez(save_path + '/GAN3_enc.npz', *lasagne.layers.get_all_param_values(encoder))
        np.savez(save_path + '/GAN3_dec.npz', *lasagne.layers.get_all_param_values(decoder))

    return encoder, decoder


def generate_images(dic, indexes, data_path, load_path, save_path=None):
    """
    Function that generates images from inputs, the encoder and the generator.

    :param dic: Dictionnary containing the name of all the input files
    :param indexes: Array containing the indexes of the input images names
    :param data_path: Path to the dataset
    :param load_path: Path to the files containing the network parameters
    :param save_path: Path to save the generated images
    """

    # Prefix of the input images
    pre = data_path + '/input_'
    log_fn("Creating theano variables")

    # Theano variables
    x = T.tensor4('x')
    z = T.matrix('z')

    # Generating the network
    enc = image_encoder()
    with np.load(load_path + '/GAN3_enc.npz') as f:
        enc_param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(enc, enc_param_values)

    gen = generator(z)
    with np.load(load_path + '/GAN3_gen.npz') as f:
        gen_param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(gen, gen_param_values)

    dec = image_decoder()
    with np.load(load_path + '/GAN3_dec.npz') as f:
        dec_param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(dec, dec_param_values)

    log_fn("Building theano functions")
    # Encoding function
    encode_fn = theano.function(
        [x],
        lasagne.layers.get_output(enc, x)
    )

    # Generation function
    gen_fn = theano.function(
        [z],
        lasagne.layers.get_output(gen, z)
    )

    # Decoding function (only for debugging)
    # dec_fn = theano.function(
    #     [z],
    #     lasagne.layers.get_output(dec, z)
    # )

    log_fn("Fetching data")
    # Loading image's names from indexes in the dictionnary
    list_of_inputs = [pre + dic.get(key.astype(np.int32)) + '.jpg' for key in indexes]
    # Loading images in array and adding some random noize
    data = data_utils.load_data(list_of_images=list_of_inputs, size=(64, 64))
    data = add_noize(data)
    # Encoding images
    encoded = encode_fn(lasagne.utils.floatX(data)).astype(np.float32)
    # Generating images
    samples = gen_fn(encoded)
    # Decoding encoded images (only for debugging)
    # decoded = dec_fn(encoded)

    # Finding center
    center = (
        int(np.floor(data.shape[2] / 2.)),
        int(np.floor(data.shape[3] / 2.))
    )

    # Reshaping all images into (batch_size, 64, 64, 3)
    data = data.transpose((0, 2, 3, 1))
    samples = samples.transpose((0, 2, 3, 1))
    # decoded = decoded.transpose((0, 2, 3, 1))
    for index, inner in enumerate(samples):
        img = data[index]
        # Replacing the center of the images
        # with the one generated
        img[
            center[0] - 16: center[0] + 16,
            center[1] - 16: center[1] + 16, :
        ] = inner[
            center[0] - 16: center[0] + 16,
            center[1] - 16: center[1] + 16, :
        ]
        # De-normalizing the image
        img = (img + 1) * 127.5
        # Making sure everything is in the rgb range
        img = np.rint(img).astype(np.int32)
        img = np.clip(img, 0, 255)
        # Creating an image like array into uint8
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        # Displaying the images
        img.show()
        # Saving the results
        if save_path is not None:
            name = '/Generated_' + dic[index] + '.png'
            img.save(save_path + name, 'PNG')

        # Decoded images display
        # img2 = decoded[index]
        # img2 = (img2 + 1) * 127.5
        # img2 = np.rint(img2).astype(np.int32)
        # img2 = np.clip(img2, 0, 255)
        # img2 = img2.astype(np.uint8)
        # img2 = Image.fromarray(img2)

    log_fn("End of Generation")


def main(args):
    # Setting logging informations
    if args.LoggingPath is None:
        set_log(False)
    else:
        post = 'log_' + time.strftime('%m_%d_%Y_%H_%M_%S') + '_gan3.log'
        logname = '/Training_' + post
        create_logging(args.LoggingPath, logname)
        set_log(True)

    # Setting random seed and batch_size
    np.random.seed(args.seed)
    batch_size = args.BatchSize

    log_fn("Loading dictionnary")
    dic = pickle.load(open(args.DataPath + '/data.pkl', 'rb'))

    log_fn("Dictionnary length {}".format(len(dic)))

    # Prefixes of the images to load
    prefixes = ['/input_', '/img_']

    # If not training, only generates images from inputs and exits
    if args.train == 0:
        log_fn("Generating images")
        indexes = np.random.randint(0, len(dic), 10)
        generate_images(
            dic=dic,
            indexes=indexes,
            data_path=args.DataPath,
            load_path=args.LoadPath,
            save_path=args.SavePath
        )
        exit(0)

    # Theano variables
    x = T.tensor4('x')
    y = T.tensor4('y')
    z = T.matrix('z')

    log_fn("Generating networks")
    # Encoder that encodes the input in a vector of shape (batch_size, 100)
    encoder = image_encoder(x)

    # Generator and discriminator networks
    gen = generator(z)
    dis = discriminator()

    # Load old network's parameters
    if args.ContinueTraining is not None:
        log_fn("Loading networks parameters")
        with np.load(args.LoadPath + '/GAN3_enc.npz') as f:
            enc_param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(encoder, enc_param_values)

        with np.load(args.LoadPath + '/GAN3_gen.npz') as f:
            gen_param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(gen, gen_param_values)

        with np.load(args.LoadPath + '/GAN3_dis.npz') as f:
            dis_param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(dis, dis_param_values)

    # Train encoder
    if args.TrainEncoder is not None:
        log_fn("Training the encoder")
        # Decode the image encoded by the encoder
        decoder = image_decoder()
        # Load old decoder's parameters
        if args.ContinueTraining is not None:
            log_fn("Loading decoder's parameters")
            with np.load(args.LoadPath + '/GAN3_dec.npz') as f:
                dec_params_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(decoder, dec_params_values)
        # Decoder output
        decode = lasagne.layers.get_output(decoder, lasagne.layers.get_output(encoder, x))
        # Encode parameters
        encode_params = lasagne.layers.get_all_params(encoder, trainable=True)
        # Decoder parameters
        decode_params = lasagne.layers.get_all_params(decoder, trainable=True)
        # Cost function for encoder decoder
        encode_decode_cost = lasagne.objectives.squared_error(decode, x).mean()
        # Update function for encoder decoder
        encode_decode_updates = lasagne.updates.adam(
            encode_decode_cost, encode_params + decode_params, learning_rate=0.001
        )
        log_fn("Building the encoder training function")
        # Training encoder function
        train_encode = theano.function(
            [x],
            encode_decode_cost,
            updates=encode_decode_updates
        )

        # Training encoder
        encoder, decoder = training_encoder(
            batch_size=batch_size,
            dic=dic,
            prefixes=prefixes,
            data_path=args.DataPath,
            train_encode=train_encode,
            save_path=args.SavePath,
            encoder=encoder,
            decoder=decoder
        )

    # Result for the discriminator with the real data
    real = lasagne.layers.get_output(dis, y)
    # Result for the discriminator for the generated images
    fake = lasagne.layers.get_output(dis, lasagne.layers.get_output(gen, z))

    # Generator and discriminator parameters
    gen_params = lasagne.layers.get_all_params(gen, trainable=True)
    dis_params = lasagne.layers.get_all_params(dis, trainable=True)

    # Discriminator cost
    dis_cost = lasagne.objectives.binary_crossentropy(real, 0.9) + \
        lasagne.objectives.binary_crossentropy(fake, 0.)

    dis_cost = dis_cost.mean()

    # Discriminator updates
    updates = lasagne.updates.adam(
        dis_cost, dis_params, learning_rate=0.0002, beta1=0.5
    )

    # Function that computes the squared error between the generated images and the real one
    # This is only use for monitoring purposes and is not used in the training algorithms
    square_error = lasagne.objectives.squared_error(lasagne.layers.get_output(gen, z), y).mean()

    log_fn("Building functions")
    # Discriminator training function
    train_fn = theano.function(
        [y, z],
        [(real > 0.5).mean(),
         (fake < 0.5).mean(),
         square_error],
        updates=updates
    )

    # Generator cost
    gen_cost = lasagne.objectives.binary_crossentropy(fake, 1).mean()

    # Generator updates
    gen_updates = lasagne.updates.adam(
        gen_cost, gen_params, learning_rate=0.0002
    )

    # Train generator function
    train_gen_fn = theano.function(
        [y, z],
        [(real > 0.5).mean(),
         (fake < 0.5).mean(),
         square_error],
        updates=gen_updates
    )

    # Encoder function
    encode_fn = theano.function(
        [x],
        lasagne.layers.get_output(encoder, x)
    )

    # Number of iterations before end of each epoch
    num_iter = np.round((len(dic) - 1) / batch_size).astype(np.int)

    log_fn("Training GAN")
    # Iterate over the epochs
    for epoch in range(args.epochs):
        it = 1  # Iteration count
        loss = 0.
        # Iterate over all the data, loading 10
        # batches at a time on ram
        for data in data_utils.load_data_to_ram(
                length=10 * batch_size,
                dic=dic,
                prefixes=prefixes,
                data_path=args.DataPath,
                size=[(64, 64), (64, 64)]
        ):
            # Iterate over each batches fo make the computations
            for batch in data_utils.minibatch_iterator(x=data[0], y=data[1], batch_size=batch_size):

                # Devide batch in inputs and targets
                inputs, targets = batch

                # Add noise to data
                inputs = add_noize(inputs)

                # Define everything as float32 for theano computations
                inputs = inputs.astype(np.float32)
                targets = targets.astype(np.float32)

                # Encode data
                n = encode_fn(inputs)

                # Training discriminator
                tmp_loss = np.array(train_fn(targets, n))

                # Training generator
                tmp_loss += np.array(train_gen_fn(targets, n))
                tmp_loss += np.array(train_gen_fn(targets, n))

                # Compute loss
                loss += tmp_loss / 3

                # Logging every now and then
                if (it % 1) == 0:
                    log_fn(
                        "Epoch: {} of {}, it {} of {}, Loss : {}".format(
                            epoch + 1, args.epochs,
                            it, num_iter, loss / it)
                    )

                # If something goes wrong and a nan appear, stop training
                if np.isnan(loss[2]):
                    log_fn("Add to break")
                    break

                it += 1  # Update iteration

            # Save parameters after every batch, just in case
            np.savez(args.SavePath + '/GAN3_gen.npz', *lasagne.layers.get_all_param_values(gen))
            np.savez(args.SavePath + '/GAN3_dis.npz', *lasagne.layers.get_all_param_values(dis))

    log_fn("End")
    return 0


def pars_args():
    """
    Parse every arguments given through the command line.

    :return arguments: Object with all arguments parsed
    """

    # Creating an argument parser
    arguments = argparse.ArgumentParser()

    # Adding arguments needed by parser
    arguments.add_argument(
        '-s', '--seed', type=int, default=32,
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
        '-logp', "--LoggingPath", type=str, default=None,
        help="Complete path to directory where to load info"
    )
    arguments.add_argument(
        '-te', "--TrainEncoder", type=str, default=None,
        help="If something is given, trains the encoder"
    )
    arguments.add_argument(
        '-ct', "--ContinueTraining", type=str, default=None,
        help="If something is give, loads param from LoadPath variables before training"
    )

    return arguments.parse_args()


if __name__ == "__main__":
    main(pars_args())
