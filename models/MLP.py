import numpy as np
import theano
import theano.tensor as T

class Hidden_Layer(object):
    """
    Hidden Layer class
    Builds a layer for
    Reference:
        - Deep Learning Tutorial
          Multilayer Perceptron
          http://deeplearning.net/tutorial/mlp.html#mlp
    """
    def __init__(self, rng, layer_input, features, noutput, W=None, b=None,
                 activation=T.tanh):

        self.input = layer_input

        if W is None:
            weight = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (features + noutput)),
                    high=np.sqrt(6. / (features + noutput)),
                    size=(features, noutput)
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=weight, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((noutput,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(layer_input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
