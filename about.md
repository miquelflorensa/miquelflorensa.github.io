<!-------------------------------------------------------------------
File:         tutorial.md
Description:  FNN tutorial with 1D data
Authors:      Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet
Created:      March 04, 2023
Updated:      March 04, 2023
Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com & james.goulet@polymtl.ca
Copyright (c) 2023 Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
-------------------------------------------------------------------->

# About cuTAGI

## What is TAGI?

TAGI stands for Tractable Approximate Gaussian Inference. It is an analytical method for performing Bayesian inference in neural networks. It can estimate the posterior mean vector and diagonal covariance matrix for weights and biases without using backpropagation.

Bayesian neural networks are neural networks that treat their parameters as random variables with prior distributions. They can provide uncertainty estimates for their predictions, which can be useful for applications such as safety-critical systems or active learning.

However, exact Bayesian inference in neural networks is computationally intractable due to the high dimensionality of the parameter space and the nonlinearity of the activation functions. Therefore, approximate methods are needed to infer the posterior distribution of the parameters given some observed data.

One common approach is to use variational inference, which minimizes a divergence measure between a variational distribution (usually chosen to be Gaussian) and the true posterior distribution. This requires using backpropagation to compute gradients of a loss function with respect to variational parameters.

TAGI is an alternative approach that does not rely on backpropagation or gradient-based optimization. Instead, it uses a series of linear transformations to map input data into latent spaces where Gaussian distributions are assumed. Then, it applies analytical formulas derived from linear algebra and probability theory to compute posterior mean vector and diagonal covariance matrix for weights and biases at each layer.

The advantages of TAGI are that it can avoid local optima, reduce overfitting, improve generalization, speed up training time, save memory usage, and handle missing data. The disadvantages are that it requires some assumptions about network architecture (such as symmetric activation functions) and data distribution (such as zero-mean Gaussian noise), and it may not capture complex posterior correlations.

## What is cuTAGI?

cuTAGI is an open-source library that implements TAGI on CUDA platform. It supports various neural network architectures such as fully-connected, convolutional, transpose convolutional layers, skip connections, pooling layers, normalization layers etc.

cuTAGI also supports different tasks such as supervised learning (classification or regression), unsupervised learning (autoencoder or generative adversarial network), reinforcement learning (policy gradient or actor-critic).

cuTAGI provides several examples of applying TAGI to different problems such as MNIST digit recognition, CIFAR-10 image classification, CelebA face generation, CartPole balancing, LunarLander landing etc.

cuTAGI aims to provide a user-friendly interface for building Bayesian neural networks with TAGI method on GPU devices. It also allows users to compare TAGI with other methods such as backpropagation or dropout.

lorem ipsum dolor sit amet ...