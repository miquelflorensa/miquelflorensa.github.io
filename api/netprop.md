# The NetProp class

The `NetProp` class is a base class for network properties defined in the backend C++/CUDA layer. It provides various attributes and methods for defining network architecture and properties.

<a href="https://github.com/miquelflorensa/cuTAGI/blob/main/pytagi/tagi_network.py" class="github-link">
  <div class="github-icon-container">
    <img src="../images/GitHub-Mark.png" alt="GitHub" height="32" width="64">
  </div>
  <div class="github-text-container">
    Github Source code
  </div>
</a>

## Attributes

- `layers`: A list containing different [layers](api/netprop?id=layer-code) of the network architecture.
- `nodes`: A list containing the number of hidden units for each layer.
- `kernels`: A list containing the kernel sizes for convolutional layers.
- `strides`: A list containing the strides for convolutional layers.
- `widths`: A list containing the widths of the images.
- `heights`: A list containing the heights of the images.
- `filters`: A list containing the number of filters (depth of image) for each layer.
- `activation`: A list containing the [activation](api/netprop?id=activation-code) function for each layer.
- `pads`: A list containing the padding applied to the images.
- `pad_types`: A list containing the types of padding.
- `shortcuts`: A list containing the layer indices for residual networks.
- `mu_v2b`: A NumPy array representing the mean of the observation noise squared.
- `sigma_v2b`: A NumPy array representing the standard deviation of the observation noise squared.
- `sigma_v`: A float representing the observation noise.
- `decay_factor_sigma_v`: A float representing the decaying factor for sigma v (default value: 0.99).
- `sigma_v_min`: A float representing the minimum value of the observation noise (default value: 0.3).
- `sigma_x`: A float representing the input noise noise.
- `is_idx_ud`: A boolean indicating whether or not to update only hidden units in the output layers.
- `is_output_ud`: A boolean indicating whether or not to update the output layer.
- `last_backward_layer`: An integer representing the index of the last layer whose hidden states are updated.
- `nye`: An integer representing the number of observations for hierarchical softmax.
- `noise_gain`: A float representing the gain for biases parameters relating to noise's hidden states.
- `noise_type`: A string indicating whether the noise is homoscedastic or heteroscedastic.
- `batch_size`: An integer representing the number of batches of data.
- `input_seq_len`: An integer representing the sequence length for LSTM inputs.
- `output_seq_len`: An integer representing the sequence length for the outputs of the last layer.
- `seq_stride`: An integer representing the spacing between sequences for the LSTM layer.
- `multithreading`: A boolean indicating whether or not to run parallel computing using multiple threads.
- `collect_derivative`: A boolean indicating whether to enable the derivative computation mode.
- `is_full_cov`: A boolean indicating whether to enable the full covariance mode.
- `init_method`: A string representing the initialization method, e.g., He and Xavier.
- `device`: A string indicating either "cpu" or "cuda".
- `ra_mt`: A float representing the momentum for the normalization layer.

## Example

```python
from pytagi import NetProp

class RegressionMLP(NetProp):
    """Multi-layer perceptron for regression task"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = [1, 1, 1, 1]
        self.nodes = [13, 50, 50, 1]
        self.activations = [0, 4, 4, 0]
        self.batch_size = 10
        self.sigma_v = 0.3
        self.sigma_v_min: float = 0.3
        self.device = "cpu"
```

## Layer Code
The following layer codes are used to represent different types of layers in the network:

- 1: Fully-connected layer
- 2: Convolutional layer
- 21: Transpose convolutional layer
- 3: Max pooling layer (currently not supported)
- 4: Average pooling
- 5: Layer normalization
- 6: Batch normalization
- 7: LSTM layer

## Activation Code
The following activation codes are used to represent different activation functions:

- 0: No activation
- 1: Tanh
- 2: Sigmoid
- 4: ReLU
- 5: Softplus
- 6: Leakyrelu
- 7: Mixture ReLU
- 8: Mixture bounded ReLU
- 9: Mixture sigmoid
- 10: Softmax with local linearization
- 11: Remax
- 12: Hierarchical softmax
