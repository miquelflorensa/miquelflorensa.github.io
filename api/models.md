## Regression MLP class

The model will have 1 input layer, 1 hidden layer and 1 output layer. The input layer will have 1 neuron, the hidden layer will have 50 neurons and the output layer will have 1 neuron. The activation function of the hidden layer will be ReLU and the batch size will be 4. The observation noise's standard deviation and its minimum will be 0.06.

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