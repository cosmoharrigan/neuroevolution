'''Evaluate a single recurrent neural network policy
'''

from nn_utilities import update_model_weights
from policy import CNNRNNPolicy
from torcs_controller import pyclient
import numpy as np


weights_index = -3

# Specify the path to the pretrained convolutional neural network weights here:
pretrained_weights_cnn = \
    np.loadtxt('experiments/train_cnn_ga_11/train_cnn_ga_mutpb_0_2.out')

saved_weights_rnn_multiple = np.loadtxt('history.out')
saved_weights_rnn_best = np.loadtxt('weights.out')

POLICY = CNNRNNPolicy(cnn_weights=pretrained_weights_cnn)

weights = saved_weights_rnn_multiple[weights_index]

update_model_weights(POLICY.rnn, weights)

# Instantiate the simulator environment
TORCS = pyclient.TORCS(episode_length=500,
                       policy=POLICY,
                       sessionid=1)

# Run an episode on the track and its mirror image and measure the average
# fitness
fitness1 = TORCS.run(track_id=1)
fitness2 = TORCS.run(track_id=2)

fitness = min(fitness1, fitness2)

print('fitness 1: {}, fitness 2: {}, min fitness: {}'.format(
    fitness1, fitness2, fitness))

import IPython; IPython.embed()
