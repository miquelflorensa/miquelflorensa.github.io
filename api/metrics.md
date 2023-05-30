# metric.py

Measure the accuracy of the prediction.

<a href="https://github.com/miquelflorensa/cuTAGI/blob/main/pytagi/metric.py" class="github-link">
  <div class="github-icon-container">
    <img src="../images/GitHub-Mark.png" alt="GitHub" height="32" width="64">
  </div>
  <div class="github-text-container">
    Github Source code
  </div>
</a>


## *mse* method

```python
def mse(prediction: np.ndarray, observation: np.ndarray) -> float:
    """Mean squared error"""
```

> Calculates the mean squared error between the prediction and observation arrays.

**Parameters**
- `prediction` (numpy.ndarray): Array containing the predicted values.
- `observation` (numpy.ndarray): Array containing the observed values.

**Returns**
- `float`: Mean squared error.

## *log_likelihood* method

```python
def log_likelihood(prediction: np.ndarray, observation: np.ndarray, std: np.ndarray) -> float:
    """Compute the averaged log-likelihood"""
```

> Calculates the averaged log-likelihood between the prediction and observation arrays.

**Parameters**
- `prediction` (numpy.ndarray): Array containing the predicted values.
- `observation` (numpy.ndarray): Array containing the observed values.
- `std` (numpy.ndarray): Array containing the standard deviations.

**Returns**
- `float`: Averaged log-likelihood.

## *rmse* method

```python
def rmse(prediction: np.ndarray, observation: np.ndarray) -> None:
    """Root mean squared error"""
```

> Calculates the root mean squared error between the prediction and observation arrays.

**Parameters**
- `prediction` (numpy.ndarray): Array containing the predicted values.
- `observation` (numpy.ndarray): Array containing the observed values.

## *classification_error* method

```python
def classification_error(prediction: np.ndarray, label: np.ndarray) -> None:
    """Compute the classification error"""
```

> Computes the classification error between the prediction and label arrays.

**Parameters**
- `prediction` (numpy.ndarray): Array containing the predicted values.
- `label` (numpy.ndarray): Array containing the true labels.
