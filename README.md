# Fourier Feature Networks | TensorFlow 2

This is a simple TensorFlow 2 implementation of Fourier Feature Networks, as
described in [this paper](https://arxiv.org/abs/2006.10739). Also see their
[project page](https://people.eecs.berkeley.edu/~bmild/fourfeat/).

The code here generates random Fourier features to use in a compositional
pattern producing network (this is the 2D image regression task described in
the paper.)

The `config.yaml` file controls hyperparameters, which are kept at their
defaults. The data is 6 images taken from [`places365`](http://places2.csail.mit.edu/).

## Visualization

The top row is a standard compositional pattern producing network, and the
second row is the same network using random Fourier features.

![](visualization.gif)
