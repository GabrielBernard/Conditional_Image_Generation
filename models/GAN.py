import numpy as np
import theano.tensor as T
import theano
import lasagne

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



class GAN(object):
    def __init__(self):
