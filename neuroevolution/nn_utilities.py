'''Utility functions to flatten the parameters of a neural network model or
update the weights of a neural network model from a flat parameter vector.
'''

import numpy as np


def flatten_parameters(model):
    """
    Gets the weights from the network layers and puts them in one flat
    parameter vector
    """
    return np.concatenate([layer.flatten() for layer in model.get_weights()])


def update_model_weights(model, new_weights):
    """
    Updates the network with new weights after they have been stored in one
    flat parameter vector
    """
    accum = 0
    for layer in model.layers:
        current_layer_weights_list = layer.get_weights()
        new_layer_weights_list = []
        for layer_weights in current_layer_weights_list:
            layer_total = np.prod(layer_weights.shape)
            new_layer_weights_list.append(
                new_weights[accum:accum + layer_total].
                    reshape(layer_weights.shape))
            accum += layer_total
        layer.set_weights(new_layer_weights_list)
