<!-------------------------------------------------------------------
File:         tutorial.md
Description:  FNN tutorial with 1D data
Authors:      Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet
Created:      March 04, 2023
Updated:      March 04, 2023
Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com & james.goulet@polymtl.ca
Copyright (c) 2023 Miquel Florensa & Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
-------------------------------------------------------------------->

# py/cuTAGI Documentation

> cuTAGI is an open-source Bayesian neural networks library that is based on Tractable Approximate Gaussian Inference (TAGI) theory. It supports various neural network architectures such as full-connected, convolutional, and transpose convolutional layers, as well as skip connections, pooling and normalization layers. cuTAGI is capable of performing different tasks such as supervised, unsupervised, and reinforcement learning. This library has a python API called pyTAGI that allows users to easily use the C++ and CUDA libraries.


## Getting Started

To get started with using our library, check out our:

- [installation guide](guide/install.md) for Windows, MacOS, and Linux (CPU + GPU).
- [quick tutorial](guide/quick-tutorial.md) for a 1D toy problem.

## Examples

In this section, you will find a series of [examples](examples/examples.md) for each available architecture that you can use as a starting point.

## API

Check out our [API reference](api/api.md) for a complete list of all the functions and classes in our library.

## Modules

pyTAGI already includes a set of modules that allow users to make their own models. Check out our [modules reference](modules/modules.md) for a list of classes and functions.

## Contributing

We welcome contributions from the community by  1) forking the project, 2) Create a feature branch, and 3) Commit your changes.

## Support

If you run into any issues or have any questions, please [open an issue](https://github.com/lhnguyen102/cuTAGI/issues) or contact us at *luongha.nguyen@gmail.com* or *james.goulet@polymtl.ca*.

## Citation

```
@misc{cutagi2022,
  Author = {Luong-Ha Nguyen and James-A. Goulet},
  Title = {cu{TAGI}: a {CUDA} library for {B}ayesian neural networks with Tractable Approximate {G}aussian Inference},
  Year = {2022},
  journal = {GitHub repository},
  howpublished = {https://github.com/lhnguyen102/cuTAGI}
}
```

## References 

* [Tractable approximate Gaussian inference for Bayesian neural networks](https://www.jmlr.org/papers/volume22/20-1009/20-1009.pdf) (James-A. Goulet, Luong-Ha Nguyen, and Said Amiri. JMLR, 2021) 
* [Analytically tractable hidden-states inference in Bayesian neural networks](https://www.jmlr.org/papers/volume23/21-0758/21-0758.pdf) (Luong-Ha Nguyen and James-A. Goulet. JMLR, 2022)
* [Analytically tractable inference in deep neural networks](https://arxiv.org/pdf/2103.05461.pdf) (Luong-Ha Nguyen and James-A. Goulet. ArXiv 2021)
* [Analytically tractable Bayesian deep Q-Learning](https://arxiv.org/pdf/2106.11086.pdf) (Luong-Ha Nguyen and James-A. Goulet. ArXiv, 2021)


## License

cuTAGI is licensed under the [MIT License](https://github.com/lhnguyen102/cuTAGI/blob/main/LICENSE).