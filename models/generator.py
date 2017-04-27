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
# import discriminator
import PIL.Image as Image

try:
    from models.utils import data_utils
except ImportError:
    from utils import data_utils


def log_fn(s):
    logging.info(s)


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


def image_encoder(input_var=None):

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

    log_fn("Discriminator output shape : {}".format( net.output_shape))

    return net


def generator(input_var=None):
    """
    Function that build the generator network
    :param input_var: Input variable that goes in Lasagne.layers.InputLayer
    """

    net = InputLayer(shape=(None, 100), input_var=input_var)
    # net = BatchNormLayer(DenseLayer(net, 1024))
    # Project
    net = BatchNormLayer(DenseLayer(net, 1024*4*4))
    # Reshape
    net = BatchNormLayer(ReshapeLayer(net, ([0], 1024, 4, 4)))
    # 512 units of 8 x 8
    net = BatchNormLayer(Deconv2DLayer(net, 512, 2, stride=2, nonlinearity=rectify))
    # net = BatchNormLayer(Conv2DLayer(net, 512, 3, pad=1, nonlinearity=rectify))
    # 256 units of 16 x 16
    net = BatchNormLayer(Deconv2DLayer(net, 256, 2, stride=2, nonlinearity=rectify))
    # net = BatchNormLayer(Conv2DLayer(net, 256, 3, pad=1, nonlinearity=rectify))
    # 128 units of 16 x 16
    net = BatchNormLayer(Conv2DLayer(net, 128, 3, pad=1, nonlinearity=rectify))
    # 3 units of 32 x 32
    net = Deconv2DLayer(net, 3, 2, stride=2, nonlinearity=tanh)

    log_fn("Generator output shape: {}".format(net.output_shape))
    return net


def generator_op(input_var=None):
    Normal = lasagne.init.Normal
    gen = InputLayer(shape=(None, 100), input_var=input_var)
    gen = BatchNormLayer(DenseLayer(gen, num_units=1024 * 4 * 4, W=Normal(0.05), nonlinearity=rectify))
    gen = ReshapeLayer(gen, ([0], 1024, 4, 4))
    gen = BatchNormLayer(Deconv2DLayer(gen, num_filters=512, filter_size=(2, 2), stride=2, W=Normal(0.05), nonlinearity=rectify))
    gen = BatchNormLayer(Deconv2DLayer(gen, num_filters=256, filter_size=(2, 2), stride=2, W=Normal(0.05), nonlinearity=rectify))
    gen = Deconv2DLayer(gen, num_filters=3, filter_size=(2, 2), stride=2, W=Normal(0.05), nonlinearity=tanh)
    # gen = Deconv2DLayer(gen, num_filters=3, filter_size=(2, 2), stride=2, nonlinearity=tanh)

    return gen


def create_logging(LoggingPath, logname):

    if not os.path.isdir(LoggingPath):
        os.mkdir(LoggingPath)

    logpath = LoggingPath + logname
    logging.basicConfig(filename=logpath,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')


def generate_images(dic, indexes, data_path, load_path):
    pre = data_path + '/input_'
    log_fn("Creating theano variables")
    x = T.tensor4('x')
    z = T.matrix('z')

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
    encode_fn = theano.function(
        [x],
        lasagne.layers.get_output(enc, x)
    )

    gen_fn = theano.function(
        [z],
        lasagne.layers.get_output(gen, z)
    )

    dec_fn = theano.function(
        [z],
        lasagne.layers.get_output(dec, z)
    )

    log_fn("Fetching data")
    list_of_inputs = [pre + dic.get(key.astype(np.int32)) + '.jpg' for key in indexes]
    data = data_utils.load_data(list_of_images=list_of_inputs, size=(64, 64))
    encoded = encode_fn(lasagne.utils.floatX(data)).astype(np.float32)
    samples = gen_fn(encoded)
    decoded = dec_fn(encoded)

    center = (
        int(np.floor(data.shape[2] / 2.)),
        int(np.floor(data.shape[3] / 2.))
    )

    data = data.transpose((0, 2, 3, 1))
    samples = samples.transpose((0, 2, 3, 1))
    decoded = decoded.transpose((0, 2, 3, 1))
    for index, inner in enumerate(samples):
        img = data[index]
        img2 = decoded[index]
        img[
        center[0] - 16: center[0] + 16,
        center[1] - 16: center[1] + 16, :
        ] = inner
        img = (img + 1) * 127.5
        img2 = (img2 + 1) * 127.5
        img = np.rint(img).astype(np.int32)
        img2 = np.rint(img2).astype(np.int32)
        img = np.clip(img, 0, 255)
        img2 = np.clip(img2, 0, 255)
        img = img.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        img = Image.fromarray(img)
        img2 = Image.fromarray(img2)
        img2.show()

    log_fn("End of Generation")


def main(args):
    if args.LoggingPath is None:
        log = True
    else:
        args.LoggingPath = os.path.expandvars(args.LoggingPath)
        post = 'log_' + time.strftime('%m_%d_%Y_%H_%M_%S') + '_gan3.log'
        logname = '/Training_' + post
        create_logging(args.LoggingPath, logname)
        log = False

    np.random.seed(args.seed)
    log_fn("Loading dictionnary")
    batch_size = args.BatchSize
    dic = pickle.load(open(args.DataPath + '/data.pkl', 'rb'))
    # index = np.arange(0, np.round(len(dic)/1000))
    # dic = {key: dic[key] for key in index}
    log_fn("Dictionnary length {}".format(len(dic)))
    prefixes = ['/input_', '/target_']

    if args.train == 0:
        indexes = np.random.randint(0, len(dic), 10)
        generate_images(
            dic=dic,
            indexes=indexes,
            data_path=args.DataPath,
            load_path=args.LoadPath
        )
        exit(0)

    x = T.tensor4('x')
    y = T.tensor4('y')
    z = T.matrix('z')

    log_fn("Generating networks")
    encoder = image_encoder(x)
    decoder = image_decoder()

    gen = generator(z)
    dis = discriminator()

    decode = lasagne.layers.get_output(decoder, lasagne.layers.get_output(encoder, x))

    encode_params = lasagne.layers.get_all_params(encoder, trainable=True)
    decode_params = lasagne.layers.get_all_params(decoder, trainable=True)

    encode_decode_cost = lasagne.objectives.squared_error(decode, x).mean()

    encode_decode_updates = lasagne.updates.adam(
        encode_decode_cost, encode_params + decode_params, learning_rate=0.001
    )

    real = lasagne.layers.get_output(dis, y)
    fake = lasagne.layers.get_output(dis, lasagne.layers.get_output(gen, z))

    gen_params = lasagne.layers.get_all_params(gen, trainable=True)
    dis_params = lasagne.layers.get_all_params(dis, trainable=True)

    # Generator
    gen_cost_dis = lasagne.objectives.binary_crossentropy(fake, 1).mean()

    # Discriminator
    dis_cost = lasagne.objectives.binary_crossentropy(real, 0.9) + \
        lasagne.objectives.binary_crossentropy(fake, 0.)

    dis_cost = dis_cost.mean()
    dis_cost_real = lasagne.objectives.binary_crossentropy(real, 1).mean()
    dis_cost_fake = lasagne.objectives.binary_crossentropy(fake, 0).mean()

    square_error = lasagne.objectives.squared_error(lasagne.layers.get_output(gen, z), y).mean()
    log_fn("Building functions")

    updates = lasagne.updates.adam(
        dis_cost, dis_params, learning_rate=0.0002, beta1=0.5
    )
    # updates = lasagne.updates.adam(
    #     gen_cost_dis, gen_params, learning_rate=0.0002, beta1=0.5
    # )

    # updates.update(lasagne.updates.adam(
    #     dis_cost, dis_params, learning_rate=0.0002, beta1=0.5
    # ))

    train_fn = theano.function(
        [y, z],
        [(real > 0.5).mean(),
         (fake < 0.5).mean(),
         square_error],
        updates=updates
    )

    # gen_cost = lasagne.objectives.squared_error(lasagne.layers.get_output(gen, z), y).mean()
    gen_cost = lasagne.objectives.binary_crossentropy(fake, 1).mean()

    gen_updates = lasagne.updates.adam(
        gen_cost, gen_params, learning_rate=0.0002
    )

    dis_updates_real = lasagne.updates.adam(
        dis_cost_real, dis_params, learning_rate=0.0002
    )

    dis_updates_fake = lasagne.updates.adam(
        dis_cost_fake, dis_params, learning_rate=0.0002
    )

    train_gen_fn = theano.function(
        [y, z],
        [(real > 0.5).mean(),
         (fake < 0.5).mean(),
         square_error],
        updates=gen_updates
    )

    train_dis_real = theano.function(
        [y, z],
        [(real > 0.5).mean(),
         (fake < 0.5).mean(),
         square_error],
        updates=dis_updates_real
    )

    train_dis_fake = theano.function(
        [y, z],
        [(real > 0.5).mean(),
         (fake < 0.5).mean(),
         square_error],
        updates=dis_updates_fake
    )

    encode_fn = theano.function(
        [x],
        lasagne.layers.get_output(encoder, x)
    )

    train_encode = theano.function(
        [x],
        encode_decode_cost,
        updates=encode_decode_updates
    )

    log_fn("Training_encoder")
    num_iter = np.round((len(dic) - 1) / batch_size).astype(np.int)
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
                data_path=args.DataPath,
                size=[(64, 64), (32, 32)]
        ):
            for batch in data_utils.minibatch_iterator(x=data[0], y=data[1], batch_size=batch_size):
                inputs, targets = batch
                inputs = add_noize(inputs)
                inputs = inputs.astype(np.float32)
                targets = targets.astype(np.float32)

                loss += train_encode(inputs)
                if (it % 1) == 0:
                    log_fn("Epoch: {} of {}, it {} of {}, Loss : {}".format(
                        epoch + 1, args.epochs,
                        it, num_iter, loss / it)
                    )

                it += 1

        np.savez(args.SavePath + '/GAN3_enc.npz', *lasagne.layers.get_all_param_values(encoder))
        np.savez(args.SavePath + '/GAN3_dec.npz', *lasagne.layers.get_all_param_values(decoder))

    decoder = None
    decode = None
    decode_params = None
    encode_decode_updates = None
    encode_decode_cost = None
    train_encode = None

    log_fn("Training GAN")
    for epoch in range(args.epochs):
        # train_both = True
        train_gen = True
        ind = 0
        it = 1
        loss = 0.
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

                n = encode_fn(inputs)
                tmp_loss = np.array(train_fn(targets, n))
                if train_gen:
                    tmp_loss = np.array(train_gen_fn(targets, n))

                loss += tmp_loss
                # if epoch == 0 and train_gen:
                #     tmp_loss = np.array(train_gen_fn(targets, n))

                # loss += tmp_loss
                # ind += min(batch_size, len(dic) - ind)
                if (it % 1) == 0:
                    log_fn("Epoch: {} of {}, it {} of {}, Loss : {}".format(
                        epoch + 1, args.epochs,
                        it, num_iter, loss / it)
                    )

                mean_real = loss[0] / it
                mean_fake = loss[1] / it
                if mean_real < 0.42 or mean_fake > 0.58:
                    train_gen = False
                    train_dis_real(targets, n)
                    train_dis_fake(targets, n)
                # if tmp_loss_gen < 0.45 or tmp_loss_dis > 0.7:
                #     train_both = False
                #     train_only_gen = True
                # elif tmp_loss_gen > 0.48 or tmp_loss_dis < 0.45:
                #     train_both = True
                #     train_only_gen = False
                # elif tmp_loss_gen > 0.6 or tmp_loss_dis < 0.45:
                #     train_both = False
                #     train_only_gen = False

                if np.isnan(loss[2]):
                    log_fn("Add to break")
                    break
                it += 1

        np.savez(args.SavePath + '/GAN3_gen.npz', *lasagne.layers.get_all_param_values(gen))
        np.savez(args.SavePath + '/GAN3_disc.npz', *lasagne.layers.get_all_param_values(dis))

    log_fn("End")
    return 0


def pars_args():
    arguments = argparse.ArgumentParser()
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
    return arguments.parse_args()


if __name__ == "__main__":
    main(pars_args())