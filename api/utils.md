# Tagi Utils 

<a href="https://github.com/miquelflorensa/cuTAGI/blob/main/pytagi/tagi_utils.py" class="github-link">
  <div class="github-icon-container">
    <img src="../images/GitHub-Mark.png" alt="GitHub" height="32" width="64">
  </div>
  <div class="github-text-container">
    Github Source code
  </div>
</a>

---

# The HierarchicalSoftmax class

Hierarchical softmax wrapper. Further details can be found [here](https://building-babylon.net/2017/08/01/hierarchical-softmax).

## *constructor* method

> Constructor for the HierarchicalSoftmax class.

```python
def __init__(self) -> None:
    super().__init__()
```

**Note:** The `super()` function is used to call the constructor of the base class `HrSoftmax`.

---

# The Utils class

Frontend for utility functions from C++/CUDA backend.

## Attributes

- `backend_utils`: Utility functionalities from the backend.

## *constructor* method

> Constructor for the Utils class.

```python
def __init__(self) -> None:
```

## *label_to_obs* method

```python
def label_to_obs(self, labels: np.ndarray,
                 num_classes: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Get observations and observation indices of the binary tree for classification"""

```

**Parameters**
- `labels`: Labels of the dataset as a NumPy array.
- `num_classes`: Total number of classes.

**Returns**
- `obs`: Encoded observations of the labels as a NumPy array.
- `obs_idx`: Indices of the encoded observations in the output vector as a NumPy array.
- `num_obs`: Number of encoded observations.

## *label_to_one_hot* method

```python
def label_to_one_hot(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Get the one hot encoder for each class"""

```

**Parameters**
- `labels`: Labels of the dataset as a NumPy array.
- `num_classes`: Total number of classes.

**Returns**
- `one_hot`: One hot encoder as a NumPy array.

## *load_mnist_images* method

```python
def load_mnist_images(self, image_file: str, label_file: str,
                      num_images: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load mnist dataset"""

```

**Parameters**
- `image_file`: Location of the Mnist image file.
- `label_file`: Location of the Mnist label file.
- `num_images`: Number of images to be loaded.

**Returns**
- `images`: Image dataset as a NumPy array.
- `labels`: Label dataset as a NumPy array.
- `num_images`: Total number of images.

## *load_cifar_images* method

```python
def load_cifar_images(self, image_file: str,
                      num: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load cifar dataset"""

```

**Parameters**
- `image_file`: Location of the image file.
- `num`: Number of images to be loaded.

**Returns**
- `images`: Image dataset as a NumPy array.
- `labels`: Label dataset as a NumPy array.

Here are the method signatures for the additional methods:

## *get_labels* method

```python
def get_labels(self, ma: np.ndarray, Sa: np.ndarray,
               hr_softmax: HierarchicalSoftmax, num_classes: int,
               batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert last layer's hidden state to labels"""
```

**Parameters**
- `ma`: Mean of activation units for the output layer as a NumPy array.
- `Sa`: Variance of activation units for the output layer as a NumPy array.
- `hr_softmax`: Hierarchical softmax.
- `num_classes`: Total number of classes.
- `batch_size`: Number of data in a batch.

**Returns**
- `pred`: Label prediction as a NumPy array.
- `prob`: Probability for each label as a NumPy array.

## *get_errors* method

```python
def get_errors(self, ma: np.ndarray, Sa: np.ndarray, labels: np.ndarray,
               hr_softmax: HierarchicalSoftmax, num_classes: int,
               batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert last layer's hidden state to labels"""
```

**Parameters**
- `ma`: Mean of activation units for the output layer as a NumPy array.
- `Sa`: Variance of activation units for the output layer as a NumPy array.
- `labels`: Label dataset as a NumPy array.
- `hr_softmax`: Hierarchical softmax.
- `num_classes`: Total number of classes.
- `batch_size`: Number of data in a batch.

**Returns**
- `pred`: Label prediction as a NumPy array.
- `prob`: Probability for each label as a NumPy array.

## *get_hierarchical_softmax* method

```python
def get_hierarchical_softmax(self, num_classes: int) -> HierarchicalSoftmax:
    """Convert labels to binary tree"""
```

**Parameters**
- `num_classes`: Total number of classes.

**Returns**
- `hr_softmax`: Hierarchical softmax.

## *obs_to_label_prob* method

```python
def obs_to_label_prob(self, ma: np.ndarray, Sa: np.ndarray,
                      hr_softmax: HierarchicalSoftmax,
                      num_classes: int) -> np.ndarray:
    """Convert observation to label probabilities"""
```

**Parameters**
- `ma`: Mean of activation units for the output layer as a NumPy array.
- `Sa`: Variance of activation units for the output layer as a NumPy array.
- `hr_softmax`: Hierarchical softmax.
- `num_classes`: Total number of classes.

**Returns**
- `prob`: Probability for each label as a NumPy array.

## *create_rolling_window* method

```python
def create_rolling_window(self, data: np.ndarray, output_col: np.ndarray,
                          input_seq_len: int, output_seq_len: int,
                          num_features: int,
                          stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create rolling window for time series data"""
```

**Parameters**
- `data`: Dataset as a NumPy array.
- `output_col`: Indices of the output columns as a NumPy array.
- `input_seq_len`: Length of the input sequence.
- `output_seq_len`: Length of the output sequence.
- `num_features`: Number of features.
- `stride`: Controls the number of steps for the window movements.

**Returns**
- `input_data`: Input data for neural networks in sequence as a NumPy array.
- `output_data`: Output data for neural networks in sequence as a NumPy array.

## *get_upper_triu_cov* method

```python
def get_upper_triu_cov

(self, batch_size: int, num_data: int,
                       sigma: float) -> np.ndarray:
    """Create an upper triangle covariance matrix for inputs"""
```

**Parameters**
- `batch_size`: Batch size as an integer.
- `num_data`: Number of data as an integer.
- `sigma`: Sigma value as a float.

**Returns**
- `vx_f`: Upper triangle covariance matrix for inputs as a NumPy array.

---

## *exponential_scheduler* method

```python
def exponential_scheduler(curr_v: float, min_v: float, decaying_factor: float,
                          curr_iter: float) -> float:
    """Exponentially decaying"""
```

**Parameters**
- `curr_v`: Current value as a float.
- `min_v`: Minimum value as a float.
- `decaying_factor`: Decaying factor as a float.
- `curr_iter`: Current iteration as a float.

**Returns**
- `float`: A float representing the result of the exponential decay calculation. The returned value is the maximum of `curr_v * (decaying_factor**curr_iter)` and `min_v`.

---

# The Normalizer class

Different methods to normalize the data before feeding it to neural networks.

## *constructor* method

> Constructor for the Normalizer class.

```python
def __init__(self, method: Union[str, None] = None) -> None:
```

**Parameters**
- `method`: Optional. A string representing the normalization method.

## *standardize* method

```python
def standardize(self, data: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalization where data_norm = (data - data_mean) / data_std """
```

**Parameters**
- `data`: Input data as a NumPy array.
- `mu`: Mean values of the data as a NumPy array.
- `std`: Standard deviation values of the data as a NumPy array.

**Returns**
- `np.ndarray`: Normalized data as a NumPy array.

## *unstandardize* method

```python
@staticmethod
def unstandardize(norm_data: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Transform standardized data to original space"""
```

**Parameters**
- `norm_data`: Standardized data as a NumPy array.
- `mu`: Mean values of the original data as a NumPy array.
- `std`: Standard deviation values of the original data as a NumPy array.

**Returns**
- `np.ndarray`: Unstandardized data in the original space as a NumPy array.

## *unstandardize_std* method

```python
@staticmethod
def unstandardize_std(norm_std: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Transform standardized std to original space"""
```

**Parameters**
- `norm_std`: Standardized standard deviation values as a NumPy array.
- `std`: Standard deviation values of the original data as a NumPy array.

**Returns**
- `np.ndarray`: Unstandardized standard deviation values in the original space as a NumPy array.

## *max_min_norm* method

```python
def max_min_norm(self, data: np.ndarray, max_value: np.ndarray, min_value: np.ndarray) -> np.ndarray:
    """Normalize the data between 0 and 1"""
```

**Parameters**
- `data`: Input data as a NumPy array.
- `max_value`: Maximum values of the data as a NumPy array.
- `min_value`: Minimum values of the data as a NumPy array.

**Returns**
- `np.ndarray`: Normalized data between 0 and 1 as a NumPy array.

## *max_min_unnorm* method

```python
@staticmethod
def max_min_unnorm(norm_data: np.ndarray, max_value: np.ndarray, min_value: np.ndarray) -> np.ndarray:
    """Transform max-min normalized data to original space"""
```

**Parameters**
- `norm_data`: Max-min normalized data as a NumPy array.
- `max_value`: Maximum values of the original data as a NumPy array.
- `min_value`: Minimum values of the original data as a NumPy array.

**Returns**
- `np.ndarray`: Unnormalized data in the original space as a NumPy array.

## *max_min_unnorm_std* method

```python
@staticmethod
def max_min_unnorm_std(norm_std: np.ndarray, max_value: np.ndarray, min_value: np.ndarray) -> np.ndarray:
    """Transform max-min normalized std to original space"""
```

**Parameters**
- `norm_std`: Max-min normalized standard deviation values as a NumPy array.
- `max_value`: Maximum values of the original data as a NumPy array.
- `min_value`:

 Minimum values of the original data as a NumPy array.

**Returns**
- `np.ndarray`: Unnormalized standard deviation values in the original space as a NumPy array.

## *compute_mean_std* method

```python
@staticmethod
def compute_mean_std(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sample mean and standard deviation"""
```

**Parameters**
- `data`: Input data as a NumPy array.

**Returns**
- `Tuple[np.ndarray, np.ndarray]`: A tuple containing the sample mean and standard deviation as NumPy arrays.

## *compute_max_min* method

```python
@staticmethod
def compute_max_min(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute max min values"""
```

**Parameters**
- `data`: Input data as a NumPy array.

**Returns**
- `Tuple[np.ndarray, np.ndarray]`: A tuple containing the maximum and minimum values of the data as NumPy arrays.

---

## *load_param_from_files function* method

Load parameters from CSV files and return them as an instance of the Param class.

```python
def load_param_from_files(mw_file: str, Sw_file: str, mb_file: str,
                          Sb_file: str, mw_sc_file: str, Sw_sc_file: str,
                          mb_sc_file: str, Sb_sc_file: str) -> Param:
    """Load parameters from CSV files"""
```

**Parameters**
- `mw_file`: Path to the CSV file containing mw values.
- `Sw_file`: Path to the CSV file containing Sw values.
- `mb_file`: Path to the CSV file containing mb values.
- `Sb_file`: Path to the CSV file containing Sb values.
- `mw_sc_file`: Path to the CSV file containing mw_sc values.
- `Sw_sc_file`: Path to the CSV file containing Sw_sc values.
- `mb_sc_file`: Path to the CSV file containing mb_sc values.
- `Sb_sc_file`: Path to the CSV file containing Sb_sc values.

**Returns**
- `Param`: An instance of the Param class containing the loaded parameter values.

Note: The function assumes that the CSV files have no headers.
