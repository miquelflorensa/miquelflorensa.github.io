# Models

?> **All models explained in this document can be found in the [python examples](https://github.com/lhnguyen102/cuTAGI/blob/main/python_examples/model.py) source code.**

In order to use the pyTAGI library, it is necessary to create a model class that inherits from the NetProp class. This NetProp class is essentially a wrapper from cuTAGI and is described in detail in its [Tagi Network API](api/tagi-network.md) section. Thus, it will be required to import this class.

```python
from pytagi import NetProp
```

## MLP Generic class

!> *This class is not implemented in the original code. It may be usefull to have it, or maybe not.*

This class is the base class for all the models. It can be customized as desired and it has the following arguments:

- layers: list of integers.
- nodes: list of integers.
- activations: list of integers.
- batch_size: integer.
- *sigma_v: float.
- *sigma_v_min: float.
- *noise_type: string.
- *noise_gain: float.
- *init_method: string.
- *device: string.

**Not mandatory arguments*

```python
from typing import Union

class MLP(NetProp):
    """Multi-layer perceptron"""

    def __init__(self, 
                 layers: list, 
                 nodes: list, 
                 activations: list,
                 batch_size: int, 
                 sigma_v: Union[float , None] = None, 
                 sigma_v_min: Union[float, None] = None,
                 noise_type: Union[str, None] = None,
                 noise_gain: Union[float, None] = None,
                 init_method: Union[str, None] = "He",
                 device: Union[str, None] = "cpu") -> None:
        super().__init__()
        self.layers = layers
        self.nodes = nodes
        self.activations = activations
        self.batch_size = batch_size
        if sigma_v is not None:
            self.sigma_v = sigma_v
        if sigma_v_min is not None:
            self.sigma_v_min = sigma_v_min
        if noise_type is not None:
            self.noise_type = noise_type
        if noise_gain is not None:
            self.noise_gain = noise_gain
        self.init_method: init_method
        self.device = device
```

## Regression MLP class

The regression model will have 1 input layer, 1 hidden layer and 1 output layer. The input layer will have 1 neuron, the hidden layer will have 50 neurons and the output layer will have 1 neuron. The activation function of the hidden layer will be ReLU and the batch size will be 4. The observation noise's standard deviation and its minimum will be 0.06.

```python
# Model
from pytagi import NetProp

class RegressionMLP(NetProp):
    """Multi-layer perceptron for regression task"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = [1, 1, 1]
        self.nodes = [1, 50, 1]
        self.activations = [0, 4, 0]
        self.batch_size = 4
        self.sigma_v = 0.06
        self.sigma_v_min: float = 0.06
        self.device = "cpu"
```

## Heteroscedastic Regression MLP class

```python
class HeterosMLP(NetProp):
    """Multi-layer preceptron for regression task where the
    output's noise varies overtime"""

    def __init__(self) -> None:
        super().__init__()
        self.layers: list = [1, 1, 1, 1]
        self.nodes: list = [1, 100, 100, 2]  # output layer = [mean, std]
        self.activations: list = [0, 4, 4, 0]
        self.batch_size: int = 10
        self.sigma_v: float = 0
        self.sigma_v_min: float = 0
        self.noise_type: str = "heteros"
        self.noise_gain: float = 1.0
        self.init_method: str = "He"
        self.device: str = "cpu"
```

## Derivative Regression MLP class

```python
class DervMLP(NetProp):
    """Multi-layer perceptron for computing the derivative of a 
    regression task"""

    def __init__(self) -> None:
        super().__init__()
        self.layers: list = [1, 1, 1, 1]
        self.nodes: list = [1, 64, 64, 1]
        self.activations: list = [0, 1, 4, 0]
        self.batch_size: int = 10
        self.sigma_v: float = 0.3
        self.sigma_v_min: float = 0.1
        self.decay_factor_sigma_v: float = 0.99
        self.collect_derivative: bool = True
        self.init_method: str = "He"
```

## MNIST Classification MLP class

```python
class MnistMLP(NetProp):
    """Multi-layer perceptron for mnist classificaiton.

    NOTE: The number of hidden states for last layer is 11 because
    TAGI use the hierarchical softmax for the classification task. 
    Further details can be found in 
    https://www.jmlr.org/papers/volume22/20-1009/20-1009.pdf
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = [1, 1, 1, 1]
        self.nodes = [784, 100, 100, 11]
        self.activations = [0, 4, 4, 0]
        self.batch_size = 10
        self.sigma_v = 1
        self.is_idx_ud = True
        self.multithreading = True
        self.device = "cpu"
```

## LSTM for Time Series Forecasting

```python
class TimeSeriesLSTM(NetProp):
    """LSTM for time series forecasting"""

    def __init__(self,
                 input_seq_len: int,
                 output_seq_len: int,
                 seq_stride: int = 1,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers: list = [1, 7, 7, 1]
        self.nodes: list = [1, 5, 5, 1]
        self.activations: list = [0, 0, 0, 0]
        self.batch_size: int = 10
        self.input_seq_len: int = input_seq_len
        self.output_seq_len: int = output_seq_len
        self.seq_stride: int = seq_stride
        self.sigma_v: float = 2
        self.sigma_v_min: float = 0.3
        self.decay_factor_sigma_v: float = 0.95
        self.multithreading: bool = False
        self.device: str = "cpu"
```

## MNIST Encoder

```python
class MnistEncoder(NetProp):
    """Encoder network for Mnist example"""

    def __init__(self) -> None:
        super().__init__()
        self.layers: list = [2, 2, 6, 4, 2, 6, 4, 1, 1]
        self.nodes: list = [784, 0, 0, 0, 0, 0, 0, 100, 10]
        self.kernels: list = [3, 1, 3, 3, 1, 3, 1, 1, 1]
        self.strides: list = [1, 0, 2, 1, 0, 2, 0, 0, 0]
        self.widths: list = [28, 0, 0, 0, 0, 0, 0, 0, 0]
        self.heights: list = [28, 0, 0, 0, 0, 0, 0, 0, 0]
        self.filters: list = [1, 16, 16, 16, 32, 32, 32, 1, 1]
        self.pads: list = [1, 0, 1, 1, 0, 1, 0, 0, 0]
        self.pad_types: list = [1, 0, 2, 1, 0, 2, 0, 0, 0]
        self.activations: list = [0, 4, 0, 0, 4, 0, 0, 4, 0]
        self.batch_size: int = 10
        self.is_output_ud: bool = False
        self.init_method: str = "He"
        self.device: str = "cuda"
```

## MNIST Decoder

```python
class MnistDecoder(NetProp):
    """Decoder network for Mnist example"""

    def __init__(self) -> None:
        super().__init__()
        self.layers: list = [1, 1, 21, 21, 21]
        self.nodes: list = [10, 1568, 0, 0, 784]
        self.kernels: list = [1, 3, 3, 3, 1]
        self.strides: list = [0, 2, 2, 1, 0]
        self.widths: list = [0, 7, 0, 0, 0]
        self.heights: list = [0, 7, 0, 0, 0]
        self.filters: list = [1, 32, 32, 16, 1]
        self.pads: list = [0, 1, 1, 1, 0]
        self.pad_types: list = [0, 2, 2, 1, 0]
        self.activations: list = [0, 4, 4, 4, 0]
        self.batch_size: int = 10
        self.sigma_v: float = 8
        self.sigma_v_min: float = 2
        self.is_idx_ud: bool = False
        self.last_backward_layer: int = 0
        self.decay_factor_sigma_v: float = 0.95
        self.init_method: str = "He"
        self.device: str = "cuda"
```
