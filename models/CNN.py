"""
Convolutional Neural Network class.

Author: Gabriel Bernard
Updated on: 2017-03-07
"""

import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d


class CNN(object):
    """
    CNN is a class that builds a Convolutional
    Neural Network with Theano.
    """

    class CNN_Layer(object):
        """
        CNN_Layer is a class that builds all the necessary elements
        to make a Convolutional Neural Network layer using Theano.
        """
        # TODO: Layer Class
        def __init__(self):
            print("Layer Class Initialization")


    def initialize_layers(self):
        """
        This funciton sets the default parameters for the CNN
        """


    def load_parameters(self):
        # TODO: load parameters
        print("Load parameters")

    # # # # # # # #
    # Constructor #
    # # # # # # # #
    def __init__(self, hidden_unnits=50, number_of_layers=20, minibatch_size=50,
                 save_param_file='CNN_param.dat', load_param_file=None):
        """
        Initialize the Convolutional Neural Network

        :param minibatch_size: size of the mini batch
        :param save_param_file: file to save new parameters
        :param load_param_file: file to load previous parameters
        """

        if load_param_file is None:
            # Setting hidden_units
            self.hidden_units = hidden_unnits
            # Setting number_of_layers
            self.number_of_layers = number_of_layers
            # Setting the minibatch_size
            self.minibatch_size = minibatch_size

            self.initialize_layers()
        else:
            self.load_param_file(load_param_file)

        # Defining file to save the parameters
        self.save_param_file = save_param_file


    def backpropagation(self):
        # TODO: Implement backprop approach
        print("Implement backprop approach")

    def update_parameters(self):
        # TODO: update parameters
        print("Implement parameters updates")

    def save_parameters(self):
        # TODO: save parameters
        print("Implement save parameters")

    def train(self, train_list, valid_list, minibatch_size, max_epoch):
        # TODO: Implement an early dropping approach
        print("Implement an early dropping approach")

    def load_param(self, file):
        # TODO: Implement loading parameters function
        print("Implement loading parameters funciton")
