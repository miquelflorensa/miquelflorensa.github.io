# The Param class

The `Param` class is a frontend API for weight and biases.

<a href="https://github.com/miquelflorensa/cuTAGI/blob/main/pytagi/tagi_network.py" class="github-link">
  <div class="github-icon-container">
    <img src="../images/GitHub-Mark.png" alt="GitHub" height="32" width="64">
  </div>
  <div class="github-text-container">
    Github Source code
  </div>
</a>

## Attributes

- `mw`: Mean of weight parameters (Type: `np.ndarray`)
- `Sw`: Variance of weight parameters (Type: `np.ndarray`)
- `mb`: Mean of bias parameters (Type: `np.ndarray`)
- `Sb`: Variance of bias parameters (Type: `np.ndarray`)
- `mw_sc`: Mean of weight parameters for the residual network (Type: `np.ndarray`)
- `Sw_sc`: Variance of weight parameters for the residual network (Type: `np.ndarray`)
- `mb_sc`: Mean of bias parameters for the residual network (Type: `np.ndarray`)
- `Sb_sc`: Variance of bias parameters for the residual network (Type: `np.ndarray`)

## *constructor* method

```python
def __init__(self, mw: np.ndarray, Sw: np.ndarray, mb: np.ndarray,
                Sb: np.ndarray, mw_sc: np.ndarray, Sw_sc: np.ndarray,
                mb_sc: np.ndarray, Sb_sc: np.ndarray) -> None:
    """Frontend apt for weight and biases"""
```

> Initialize an instance of the `Param` class.

**Parameters:**
- `mw` (numpy.ndarray): Mean of weight parameters.
- `Sw` (numpy.ndarray): Variance of weight parameters.
- `mb` (numpy.ndarray): Mean of bias parameters.
- `Sb` (numpy.ndarray): Variance of bias parameters.
- `mw_sc` (numpy.ndarray): Mean of weight parameters for the residual network.
- `Sw_sc` (numpy.ndarray): Variance of weight parameters for the residual network.
- `mb_sc` (numpy.ndarray): Mean of bias parameters for the residual network.
- `Sb_sc` (numpy.ndarray): Variance of bias parameters for the residual network.

**Example**

```python
from pytagi import Param
import numpy as np

mw = np.array([0.5, 0.6, 0.7])
Sw = np.array([0.1, 0.2, 0.3])
mb = np.array([0.1, 0.2])
Sb = np.array([0.01, 0.02])
mw_sc = np.array([0.8, 0.9, 1.0])
Sw_sc = np.array([0.05, 0.1, 0.15])
mb_sc = np.array([0.05, 0.1])
Sb_sc = np.array([0.005, 0.01])

param = Param(mw, Sw, mb, Sb, mw_sc, Sw_sc, mb_sc, Sb_sc)
```
