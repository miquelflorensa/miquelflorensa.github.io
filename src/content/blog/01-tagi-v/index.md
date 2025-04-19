---
title: "TAGI-V"
description: "AGVI & TAGI-V theory and implementation."
date: "Sep 13 2024"
---

*Bayesian neural networks* (BNNs) offer a probabilistic approach to estimate the model’s confidence in both its parameters and predictions. However, performing exact Bayesian Inference is NP-hard[^1], making it computationally intractable.

*Tractable Approximate Gaussian Inference* (TAGI) distinguishes itself from other methods by enabling BNNs to perform analytical parameter inference with $O(n)$ computational complexity, where $n$ is the number of parameters[^2]. But TAGI in its current form can only model homoscedastic aleatory uncertainty that is quantified by a constant error variance across the input covariate domain.

Deka et al. developed the *Approximate Gaussian Variance Inference* (AGVI), which enables analytical inference of the error variance term as a Gaussian random variable[^3]. In this context, TAGI‑V combines TAGI and AGVI to allow the modeling of heteroscedastic aleatory uncertainty in BNNs[^4].

In this article we will see how AGVI works and how to apply it on TAGI, resulting in TAGI‑V. Then we will go through the implementation of TAGI‑V in the cuTAGI[^5] open source library. In case you don’t know about TAGI, go check the last blog where we explain step by step this method.

<img src="/src/content/blog/01-tagi-v/images/heteros.jpg" alt="s" />
<figcaption class="text-center">Heteroscedastic representation, from Ideogram.</figcaption>

## AGVI

The core idea of **Approximate Gaussian Variance Inference (AGVI)** is to lift the constant error variance in TAGI into a full Gaussian random variable that can adapt per input. AGVI proceeds in two main steps:

1. **Prior via Gaussian Multiplicative Approximation (GMA)**  
   We begin by modeling the squared error $V^2$ as a Gaussian:  
   $$
   f(v^2) = \mathcal{N}\bigl(v^2;\,\mu_{V^2},\,\sigma^2_{V^2}\bigr)
   \quad\text{with}\quad \sigma^2_{V^2} = 2\,\mu_{V^2}^2.
   $$  
   Since $V$ itself is zero–mean, its variance equals $\mathbb{E}[V^2] = \mu_{V^2}$, giving  
   $$
   f(v)=\mathcal{N}(v;0,\mu_{V^2}),
   $$  
   and by treating $\mu_{V^2}$ as a Gaussian random variable, we preserve analytical tractability.

2. **Posterior update via Gaussian conditioning**  
   Once the network’s predictive distribution for the error $V$ is available through the TAGI inference pass, we update our belief on $\sigma^2_V$ by conditioning on the observed residuals.  Concretely, if the prior on $V^2$ is $\mathcal{N}(\mu_{V^2},\,2\mu_{V^2}^2)$ and the posterior on $V$ yields a mean-zero Gaussian with variance $\hat\sigma_V^2$, then the posterior moments of $V^2$ follow directly from standard Gaussian conditional formulas.

By iterating these two steps—forward propagation of moments for both the main output and the variance head, and backward conditioning—we obtain closed‑form updates for both the network parameters and the input‑dependent noise variance.

## Heteroscedasticity

In uncertainty quantification, **homoscedastic** aleatory uncertainty assumes a single noise level for all inputs,  
$$
y = z^{(\mathrm{O})} + v,\quad v\sim\mathcal{N}(0,\sigma_V^2),
$$  
where $\sigma_V^2$ is fixed[^2]. While simple, this fails to capture cases where data noise naturally varies across the domain.

By contrast, **heteroscedastic** uncertainty allows $\sigma_V^2$ to depend on the input $\mathbf{x}$:  
$$
y = z^{(\mathrm{O})}(\mathbf{x}) + v,\quad v\sim\mathcal{N}(0,\sigma_V^2(\mathbf{x})).
$$  
Modeling $\sigma_V^2(\mathbf{x})$ is crucial when different regions of the input space exhibit different noise levels—e.g., sensor readings with condition‑dependent precision. AGVI equips TAGI with exactly this capability by inferring $\sigma_V^2$ analytically rather than tuning it as a hyperparameter.

## TAGI‑V

**TAGI‑V** seamlessly integrates TAGI’s analytic parameter inference with AGVI’s variance inference, yielding a fully closed‑form, heteroscedastic Bayesian neural network.

- **Dual‑head architecture**: Each network outputs both  
  $$
  z^{(\mathrm{O})}\quad\text{and}\quad \bar V^2,
  $$  
  where $\bar V^2$ is the network’s prediction of the local noise variance.

- **Computational complexity**: TAGI‑V retains the $O(n)$ scaling in the number of parameters $n$ of original TAGI, thanks to GMA and layer‑wise recursion.

- **Empirical performance**: On standard regression benchmarks, TAGI‑V delivers superior test log‑likelihood compared to homoscedastic TAGI and matches or outperforms other approximate inference methods—while being an order of magnitude faster.

## TAGI‑V in cuTAGI

The **cuTAGI** library provides a CUDA‑accelerated implementation of both TAGI and TAGI‑V:

- **Output layout**  
  In cuTAGI’s implementation, the final layer’s tensor is of size $2Y$. Even indices hold the predictive means $z^{(\mathrm{O})}$, and odd indices hold the variance head $\bar V^2$. Internally, cuTAGI splits this tensor and runs two parallel TAGI inference tracks: one for the mean and one for the variance.

- **API usage**  
  ```python
  net = Sequential(
        Linear(1, 128),
        ReLU(),
        Linear(128, 128),
        ReLU(),
        Linear(128, 1*2), # Double of inputs
        EvenExp(),        # Exponential for V2_bar
  )

  # ...

  # Testing
  for x, y in test_batch_iter:
        # Predicion
        m_pred, v_pred = net(x)

        # Even positions correspond to Z_out and odd positions to V
        var_preds.extend(v_pred[::2] + m_pred[1::2])

        mu_preds.extend(m_pred[::2])

        x_test.extend(x)
        y_test.extend(y)
  ```


---

## References

[^1]: Cooper, G. F. (1990). *The computational complexity of probabilistic inference using Bayesian belief networks*. _Artificial Intelligence_, 42, 393–405.  
[^2]: Goulet, J. A., Nguyen, L. H., & Amiri, S. (2021). *Tractable approximate Gaussian inference for Bayesian neural networks*. _Journal of Machine Learning Research_, 22(251), 1–23.  
[^3]: Deka, B., & Goulet, J. A. (2023). *Approximate Gaussian Variance Inference for State‑Space Models*. _International Journal of Adaptive Control and Signal Processing_, doi:10.1002/acs.3456.  
[^4]: Deka, B., Nguyen, L. H., & Goulet, J. A. (2024). *Analytically tractable heteroscedastic uncertainty quantification in Bayesian neural networks*. _Neurocomputing_, 127183.  
[^5]: Nguyen, L. H., & Goulet, J. A. (2022). *cuTAGI: a CUDA library for Bayesian neural networks with tractable approximate Gaussian inference*. https://github.com/lhnguyen102/cuTAGI  
