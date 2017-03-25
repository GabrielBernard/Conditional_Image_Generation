"""
Convolutional Neural Network class.

Author: Gabriel Bernard
Updated on: 2017-03-07
"""

import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool


class CNN_Layer(object):
    """
    CNN_Layer is a class that builds all the necessary elements
    to make a Convolutional Neural Network layer using Theano.

    References:
    - Deep Learning Tutorial
      Convolutional Neural Networks (LeNet)
      http://deeplearning.net/tutorial/lenet.html#lenet
    """
    # # # # # # # #
    # Constructor #
    # # # # # # # #
    def __init__(self, rng, layer_input, filter_shape, image_shape, poolsize=(3, 3), subsample=None):
        """
        Create a Convolutional Neural Network Layer
        with shared variables to use with a GPU.

        :param rng: Random Number Generator.

        :param layer_input: T.dtensor4 representing the symbolic
                            image tensor of image_shape.

        :param filter_shape: tuple ( number of filters, number of
                             feature maps, filter height, filter width ).

        :param image_shape: tuple ( batch size, number of feature maps,
                            image height, image width ).

        :param poolsize: pooling factor.
        """

        assert image_shape[1] == filter_shape[1]
        self.input = layer_input

        #
        fan_in = np.prod(filter_shape[1:])

        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolsize))
        # Initialize weights with random values
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # Initialize the biases
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # Convolution
        conv = conv2d(
            input=layer_input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            subsample=subsample
        )

        if poolsize is not None:
            # Pool feature with max pooling
            pooled = pool.pool_2d(
                input=conv,
                ds=poolsize,
                ignore_border=True
            )
        else:
            pooled = 0

        # Calculation of the activation function with the addition of
        # the bias term. The function dimshuffle reshapes the bias to
        # ( 1, n_filter, 1, 1 ).
        self.output = T.tanh(pooled + self.b.dimshuffle('x', 0, 'x', 'x'))

        # Store parameters for later use
        self.params = [self.W, self.b]
