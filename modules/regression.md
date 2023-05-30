# Regression class

?> **This python file contains the Regression class that makes use of TAGI to train and test a regression model. The source code can be found [here](https://github.com/lhnguyen102/cuTAGI/blob/main/python_examples/regression.py).**

```python
class Regression(num_epochs=num_epochs, data_loader=data_loader, net_prop=net_prop, viz=viz)
```

## Parameters

<table>
  <tr>
    <td><span style="color:#0087ca">num_epochs</span> (int): </td>
    <td>Number of epochs to train the model</td>
  </tr>
  <tr>
    <td><span style="color:#0087ca">data_loader</span> (DataLoader): </td>
    <td>DataLoader object</td>
  </tr>
  <tr>
    <td><span style="color:#0087ca">net_prop</span> (NetProp): </td>
    <td>NetProp object</td>
  </tr>
  <tr>
    <td><span style="color:#0087ca">viz</span> (PredictionViz, default=None): &nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td>PredictionViz object</td>
  </tr>
</table>

## Methods

<table>
  <tr>
    <th>Method</th>
    <th>Parameters</th>
    <th>Returns</th>
    <th>Description</th>
  <tr>
    <td><span style="color:#0087ca">train</span>()</td>
    <td>Self</td>
    <td>None</td>
    <td>Train the model</td>
  </tr>
  <tr>
    <td><span style="color:#0087ca">predict</span>(std_factor)</td>
    <td>std_factor: int, default = 1</td>
    <td>None</td>
    <td>Test the model</td>
  </tr>
  <tr>
    <td><span style="color:#0087ca">compute_derivative</span>(layer, truth_derv_file)</td>
    <td>layer: int, default = 0<br>
    truth_derv_file = str, default = None</td>
    <td>None</td>
    <td>Compute the derivatives of the model with respect to the input</td>
  </tr>
  <tr>
    <td><span style="color:#0087ca">init_inputs</span>(batch_size)</td>
    <td>batch_size: int</td>
    <td>Tuple[np.ndarray, np.ndarray]</td>
    <td>Initialize the inputs</td>
  </tr>
  <tr>
    <td><span style="color:#0087ca">init_inputs</span> (batch_size (int)): </td>
    <td>batch_size: int</td>
    <td>Tuple[np.ndarray, np.ndarray]</td>
    <td>Initialize the outputs</td>
  </tr>
</table>
