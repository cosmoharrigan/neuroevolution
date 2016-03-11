'''Create a recurrent neural network to compute a control policy.
'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import SimpleRNN


def create_rnn():
    """Create a recurrent neural network to compute a control policy.

    Reference:
    Koutnik, Jan, Jurgen Schmidhuber, and Faustino Gomez. "Evolving deep
    unsupervised convolutional networks for vision-based reinforcement
    learning." Proceedings of the 2014 conference on Genetic and
    evolutionary computation. ACM, 2014.
    """
    model = Sequential()

    model.add(SimpleRNN(output_dim=3, stateful=True, batch_input_shape=(1, 1, 3)))
    model.add(Dense(input_dim=3, output_dim=3))

    model.compile(loss='mse', optimizer='rmsprop')

    return model


def calculate_rnn_output(model, input, multiple=False):
    """Calculates the output of the RNN. Use vector=False to indicate that a
    single input is being passed.
    """
    output = model.predict(input)
    if multiple:
        output = output.reshape(output.shape[0], output.shape[1])
    else:
        output = output.reshape(output.shape[1])

    return output
