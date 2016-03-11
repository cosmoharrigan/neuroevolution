'''Import the dataset containing the training images for the convolutional
neural network
'''

import numpy as np
import scipy.ndimage
import scipy


def load_images_torcs_4():
    import glob
    filepath = 'datasets/torcs_4/training_set/*.png'
    filenames = glob.glob(filepath)

    sample = scipy.ndimage.imread(filenames[0])
    num_images = len(filenames)
    images = np.zeros((num_images, sample.shape[0], sample.shape[1]), dtype=np.uint8)

    for i in range(num_images):
        images[i] = scipy.ndimage.imread(filenames[i])

    images = images.reshape(len(images), 1, 64, 64)

    return images
