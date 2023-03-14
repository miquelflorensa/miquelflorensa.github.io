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

## Train Algorithm

```python
train():
    # Inputs
    batch_size = net_prop.batch_size
    Sx_batch, Sx_f_batch = init_inputs(batch_size)

    # Outputs
    V_batch, ud_idx_batch = init_outputs(batch_size)

    # Load data
    input_data, output_data = data_loader["train"]
    num_data = input_data.shape[0]
    num_iter = num_data / batch_size

    # Epochs
    for epoch in range(num_epochs):
        # Update observation's variance
        sigma_v = exponential_scheduler(
            curr_v=net_prop.sigma_v,
            min_v=net_prop.sigma_v_min,
            decaying_factor=net_prop.decay_factor_sigma_v,
            curr_iter=epoch)
        V_batch = V_batch * 0.0 + sigma_v**2

        # Iterations
        for i in range(num_iter):
            # Get data randomly
            idx = random_choice(num_data, batch_size)
            x_batch = input_data[idx, :]
            y_batch = output_data[idx, :]

            # Feed forward step
            network.feed_forward(x_batch, Sx_batch, Sx_f_batch)

            # Update hidden states
            network.state_feed_backward(y_batch, V_batch,
                                         ud_idx_batch)

            # Update parameters
            network.param_feed_backward()

            # Loss
            norm_pred, _ = network.get_network_predictions()
            pred = normalizer.unstandardize(
                norm_data=norm_pred,
                mu=data_loader["y_norm_param_1"],
                std=data_loader["y_norm_param_2"])
            obs = normalizer.unstandardize(
                norm_data=y_batch,
                mu=data_loader["y_norm_param_1"],
                std=data_loader["y_norm_param_2"])
            mse = metric.mse(pred, obs)
```
## Predict Algorithm

```python
def predict(self, std_factor: int = 1) -> None:
    """Make prediction using TAGI"""
    
    # Get batch size and initialize inputs
    batch_size = self.net_prop.batch_size
    Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

    # Initialize empty lists to store predictions and actual values
    mean_predictions = []
    variance_predictions = []
    y_test = []
    x_test = []

    # Iterate through test data in batches
    for x_batch, y_batch in self.data_loader["test"]:
        # Make predictions using feed forward
        self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)
        ma, Sa = self.network.get_network_predictions()

        # Append mean and variance predictions, as well as test inputs and outputs
        mean_predictions.append(ma)
        variance_predictions.append(Sa + self.net_prop.sigma_v**2)
        x_test.append(x_batch)
        y_test.append(y_batch)

    # Flatten and stack predictions and actual values
    mean_predictions = np.stack(mean_predictions).flatten()
    std_predictions = (np.stack(variance_predictions).flatten())**0.5
    y_test = np.stack(y_test).flatten()
    x_test = np.stack(x_test).flatten()

    # Unnormalize the data
    mean_predictions = normalizer.unstandardize(
        norm_data=mean_predictions,
        mu=self.data_loader["y_norm_param_1"],
        std=self.data_loader["y_norm_param_2"])
    std_predictions = normalizer.unstandardize_std(
        norm_std=std_predictions, std=self.data_loader["y_norm_param_2"])

    x_test = normalizer.unstandardize(
        norm_data=x_test,
        mu=self.data_loader["x_norm_param_1"],
        std=self.data_loader["x_norm_param_2"])
    y_test = normalizer.unstandardize(
        norm_data=y_test,
        mu=self.data_loader["y_norm_param_1"],
        std=self.data_loader["y_norm_param_2"])

    # Compute log-likelihood, MSE, and error rate
    mse = metric.mse(mean_predictions, y_test)
    log_lik = metric.log_likelihood(prediction=mean_predictions,
                                    observation=y_test,
                                    std=std_predictions)
    error_rate = metric.error_rate(mean_predictions, y_test)
```
