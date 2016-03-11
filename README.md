## Neuroevolution as a direct policy search deep reinforcement learning method

Implemented using the [Keras](http://keras.io/) deep learning library and the [DEAP](http://deap.readthedocs.org/en/master/) evolutionary computation framework.

[Follow @cosmosquared](https://twitter.com/cosmosquared)

**Project Status:** This project is still a work in progress and is not finished.

## Overview

In direct policy search, the space of possible policies is searched directly. The agent does not attempt to model the transition dynamics of the environment, nor does it attempt to explicitly learn the value of different states or actions. Instead, it iteratively attempts to improve a parameterized policy.

Direct policy search can be broken down into gradient-based methods, also known as policy
gradient methods, and methods that do not rely on the gradient. Gradient-free methods include evolutionary algorithms.

In this architecture, a convolutional neural network acts as a compressor for images perceived by an agent in its environment. It is defined in [cnn.py](neuroevolution/cnn.py) and is trained by neuroevolution using [train_cnn_ga.py](neuroevolution/train_cnn_ga.py).

A recurrent neural network acts as a controller based on the output of the compressor. It is defined in [rnn.py](neuroevolution/rnn.py) and is trained by neuroevolution using [evolve_rnn_controller.py](neuroevolution/evolve_rnn_controller.py). A client that interacts with the environment must be defined and configured. The implementation is illustrated using a client that interacts with a customized version of the [TORCS](http://torcs.sourceforge.net/) car racing simulation.

## Prior work

This implementation is loosely based on the architecture described in:

- Koutnik, Jan, Jurgen Schmidhuber, and Faustino Gomez. "Evolving deep unsupervised convolutional networks for vision-based reinforcement learning." Proceedings of the 2014 conference on Genetic and evolutionary computation. ACM, 2014.

## Citation

```
@misc{neuroevolution,
  author = {Harrigan, Cosmo},
  title = {Neuroevolution as a direct policy search deep reinforcement learning method},
  year = {2016},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cosmoharrigan/neuroevolution}}
}
```
