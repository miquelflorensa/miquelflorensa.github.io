# The TagiNetwork class

Python frontend calling TAGI network in C++/CUDA backend.

<a href="https://github.com/miquelflorensa/cuTAGI/blob/main/pytagi/tagi_network.py" class="github-link">
  <div class="github-icon-container">
    <img src="../images/GitHub-Mark.png" alt="GitHub" height="32" width="64">
  </div>
  <div class="github-text-container">
    Github Source code
  </div>
</a>

## Attributes

- `network`: Network wrapper that calls the tagi network from the backend
- `net_prop`: Network properties

## *constructor* method

> Constructor for the TagiNetwork class.

```python
def __init__(self, net_prop: NetProp) -> None:
```

**Parameters**

- `net_prop`: An instance of the [NetProp class](api/netprop.md) representing the network properties.

## *net_prop* getter method

```python
@property
def net_prop(self) -> NetProp():
    """"Get network properties"""
```

**Returns**

- `NetProp`: An instance of the [NetProp class](api/netprop.md).

## *net_prop* setter method

```python
@net_prop.setter
def net_prop(self, value: NetProp) -> None:
    """Set network properties"""
```

**Parameters**

- `value`: An instance of the [NetProp class](api/netprop.md) class representing the network properties.

## *feed_forward* method

```python
def feed_forward(self, x_batch: np.ndarray, 
                       Sx_batch: np.ndarray,
                       Sx_f_batch: np.ndarray) -> None:
    """Forward pass the size of x_batch, Sx_batch (B, N) 
    where B is the batch size and N is the data dimension"""
```

**Parameters**

- `x_batch`: Input data as a NumPy array.
- `Sx_batch`: Diagonal variance of input data as a NumPy array.
- `Sx_f_batch`: Full variance of input data as a NumPy array.

## *connected_feed_forward* method

```python
def connected_feed_forward(self, ma: np.ndarray, va: np.ndarray,
                           mz: np.ndarray, vz: np.ndarray,
                           jcb: np.ndarray) -> None:
    """Forward pass for the network that is connected to the other 
    network e.g., decoder network in autoencoder task where its inputs 
    are the outputs of the encoder network."""
```

**Parameters**

- `ma`: Mean of activation units as a NumPy array.
- `va`: Variance of activation units as a NumPy array.
- `mz`: Mean of hidden states as a NumPy array.
- `vz`: Variance of hidden states as a NumPy array.
- `jcb`: Jacobian matrix (da/dz) as a NumPy array.

## *state_feed_backward* method

```python
def state_feed_backward(self, y_batch: np.ndarray, 
                        v_batch: np.ndarray,
                        ud_idx_batch: np.ndarray) -> None:
    """Update hidden states the size of y_batch, V_batch (B, N) 
    where B is the batch size and N is the data dimension"""
```

**Parameters**

- `y_batch`: Observations as a NumPy array.
- `v_batch`: Variance of observations as a NumPy array.
- `ud_idx_batch`: Updated indices for the last layer as a NumPy array.

## *param_feed_backward* method

> Update parameters.

```python
TagiNetwork.param_feed_backward()
```

## *get_network_outputs* method

> Get output layer's hidden state distribution.

```python
ma, va = TagiNetwork.get_network_outputs()
```

**Returns**

- `ma`: Mean of activation units as a NumPy array.
- `va`: Variance of activation units as a NumPy array.

## *get_network_predictions* method

> Get distribution of the predictions.

```python
m_pred, v_pred = TagiNetwork.get_network_predictions()
```

**Returns**

- `m_pred`: Mean of predictions as a NumPy array.
- `v_pred`: Variance of predictions as a NumPy array.

## *get_all_network_outputs* method

> Get all hidden states of the output layers.

```python
ma, va, mz, vz, jcb = TagiNetwork.get_all_network_outputs()
```

**Returns**

- `ma`: Mean of activations for the output layer as a NumPy array.
- `va`: Variance of activations for the output layer as a NumPy array.
- `mz`: Mean of hidden states for the output layer as a NumPy array.
- `vz`: Variance of hidden states for the output layer as a NumPy array.
- `jcb`: Jacobian matrix for the output layer as a NumPy array.

## *get_all_network_inputs* method

> Get all hidden states of the input layers.

```python
ma, va, mz, vz, jcb = TagiNetwork.get_all_network_inputs()
```

**Returns**

- `ma`: Mean of activations for the input layer as a NumPy array.
- `va`: Variance of activations for the input layer as a NumPy array.
- `mz`: Mean of hidden states for the input layer as a NumPy array.
- `vz`: Variance of hidden states for the input layer as a NumPy array.
- `jcb`: Jacobian matrix for the input layer as a NumPy array.

## *get_derivatives* method

> Compute derivatives of the output layer w.r.t a given layer using TAGI.

```python
mdy, vdy = TagiNetwork.get_derivatives(layer: int)
```

**Parameters**

- `layer`: Layer index of the network.

**Returns**

- `mdy`: Mean values of derivatives as a NumPy array.
- `vdy`: Variance values of derivatives as a NumPy array.

## *get_inovation_mean_var* method

> Get updating quantities for the innovation.

```python
delta_m, delta_v = TagiNetwork.get_inovation_mean_var(layer: int)
```

**Parameters**

- `layer`: Layer index of the network.

**Returns**

- `delta_m`: Innovation mean as a NumPy array.
- `delta_v`: Innovation variance as a NumPy array.

## *get_state_delta_mean_var(self)* method

> Get updating quantities for the first layer.

```python
delta_mz, delta_vz = TagiNetwork.get_state_delta_mean_var()
```

**Returns**

- `delta_mz`: Updating quantities for the hidden-state mean of the first layer as a NumPy array.
- `delta_vz`: Updating quantities for the hidden-state variance of the first layer as a NumPy array.

## *set_parameters* method

> Set parameter values to the network.

```python
TagiNetwork.set_parameters(param: Param)
```

**Parameters**

- `param`: An instance of the [Param class](api/param.md) representing the parameter values.

## *get_parameters* method

> Get parameters of the network.

```python
param = TagiNetwork.get_parameters()
```

**Returns**

- `param`: An instance of the [Param class](api/param.md) representing the network parameters.