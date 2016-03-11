'''Create a convolutional neural network for compressing the input images.
'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from scipy.spatial import distance
import sklearn.preprocessing


def create_cnn():
    """Create a convolutional neural network for compressing the input images.

    Reference:
    Koutnik, Jan, Jurgen Schmidhuber, and Faustino Gomez. "Evolving deep
    unsupervised convolutional networks for vision-based reinforcement
    learning." Proceedings of the 2014 conference on Genetic and
    evolutionary computation. ACM, 2014.
    """
    model = Sequential()

    model.add(Convolution2D(nb_filter=10, nb_row=2, nb_col=2,
                            input_shape=(1, 64, 64)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(nb_filter=10, nb_row=2, nb_col=2,
                            input_shape=(10, 21, 21)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(nb_filter=10, nb_row=2, nb_col=2,
                            input_shape=(10, 10, 10)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(nb_filter=3, nb_row=2, nb_col=2,
                            input_shape=(10, 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # The model needs to be compiled before it can be used for prediction
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


def calculate_cnn_output(model, input):
    output = model.predict(input)
    output = output.reshape(output.shape[0], output.shape[1])

    normalized_output = sklearn.preprocessing.normalize(output)

    return normalized_output


def calculate_fitness(feature_vectors):
    pairwise_euclidean_distances = distance.pdist(feature_vectors, 'euclidean')
    fitness = pairwise_euclidean_distances.mean() + \
              pairwise_euclidean_distances.min()
    return fitness
