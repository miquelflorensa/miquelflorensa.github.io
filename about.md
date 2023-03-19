<!-------------------------------------------------------------------
File:         about.md
Description:  
Authors:      Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet
Created:      March 04, 2023
Updated:      March 04, 2023
Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com & james.goulet@polymtl.ca
Copyright (c) 2023 Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
-------------------------------------------------------------------->

# About py/cuTAGI

The core developpements of py/cuTAGI have been made by Luong-Ha Nguyen building upon the theoretical work done at Polytechnique Montreal in collaboration with James-A. Goulet, Bhargob Deka, Van-Dai Vuong and Miquel Florensa. The project started in 2018 when, from our background with large-scale state-space models, we foresaw that it would be possible to perform analytical Bayesian inference in neural networks (see below our first try at what would become TAGI).
<p align="center">
<img src="./images/TAGI_2018.png"  width="40%" alt="TAGI initial trial iun 2018">
</p>
Following the early proofs of concepts with small-scale examples with single-layer MLPs, we slowly expanded the developpement of TAGI for CNN, autotoencoders and GANs architctures. Then came proofs of concepts with reinforcement learning toy problems which led to full-scale applications on the Atari and MuJoCo benchmarks. The expansion of TAGI's applicability to new architectures continued with LSTM networks along with unprecedented features with analytical uncertainty quantification for Bayesian neural networks, analytical adversaial attacks, inference-based optimization and general purpose latent-space inference. 

Despite our repeated successes at leveraging analytical inference in neural network, the key limitation remaining was the lack of a efficient and scalalable library for TAGI; as the method does not relies on Backprop nor gradient descent, it is incompatible with traditionnal libraries such as PyTorch or TensorFlow. In 2021, Luong-Ha Nguyen decided to lead the developpement of the new cuTAGI plateform and later on the pyTAGI API with the objective to open the capabilities of TAGI to the entire community.   