'''Defines a policy that calculates a control signal from an image.
'''

from cnn import create_cnn, calculate_cnn_output
from nn_utilities import update_model_weights
from rnn import create_rnn, calculate_rnn_output


class CNNRNNPolicy(object):
    def __init__(self, cnn_weights=None):
        # Create the ConvNet and load the pretrained weights if provided
        self.cnn = create_cnn()
        if cnn_weights is not None:
            update_model_weights(self.cnn, cnn_weights)
        # Create the RNN
        self.rnn = create_rnn()

    def rnn_output(self, percept):
        # Feed the image through the ConvNet to get a low dimensional feature
        # vector. Returns a vector in R3 for use in control.
        feature_vector = calculate_cnn_output(self.cnn,
                                              percept.reshape(1, 1, 64, 64))

        # Feed the feature vector into the RNN and compute a control signal
        # from its output
        rnn_out = calculate_rnn_output(self.rnn,
                                       feature_vector.reshape(1, 1, 3))

        return rnn_out

    def control_signal(self, rnn_out):
        """
        Given an output from the RNN calculate the control signals:
        acceleration/braking and turn angle

        Let rnn_output = [o1, o2, o3]. Then, we calculate:

        signal_steering = (o1 + o2) / 2
        Between -1 (full left) and +1 (full right)

        signal_acceleration = o3
        Between -1 (full brake) and +1 (full throttle)

        Returns a pair: signal_steering, signal_acceleration
        """
        signal_steering = (rnn_out[0] + rnn_out[1]) / 2
        signal_acceleration = rnn_out[2]

        # Restrict steering to the range [-1, 1]
        if signal_steering < -1:
            signal_steering = -1
        elif signal_steering > 1:
            signal_steering = 1

        # Restrict acceleration to the range [0, 1]
        if signal_acceleration < 0:
            signal_acceleration = 0
        elif signal_acceleration > 1:
            signal_acceleration = 1

        return signal_steering, signal_acceleration
