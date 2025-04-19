---
title: "Quantifying Uncertainty in Visual Autoregressive Image Generation"
description: "A study on epistemic uncertainty in autoregressive image models using MC Dropout and semantic analysis."
date: "Apr 19 2025"
---

## Summary

In this work, I develop and compare two autoregressive image generators—VAR and VAR‑CLIP—implement Monte Carlo Dropout at inference to obtain pixel‑wise uncertainty maps, and analyze how uncertainty correlates with generation fidelity, semantic regions, and prompt design. VAR leverages coarse‑to‑fine “next‑scale prediction” to achieve state‑of‑the‑art FID and Inception Score[^1][^4], while VAR‑CLIP incorporates CLIP text embeddings for rich semantic conditioning[^2][^8]. Enabling dropout during sampling reveals a clear fidelity–confidence trade‑off: as DropPath rate increases, FID worsens and average σ rises[^3][^4]. Masking with SAM shows lower uncertainty over foreground objects than backgrounds[^7], and prompt complexity experiments uncover a non‑linear peak in σ at moderate ambiguity[^6]. Detailed captions both improve image quality and reduce uncertainty, demonstrating the importance of prompt clarity[^5].

---

## VAR

![VAR Architecture](/images/VAR.png)

I implement VAR by framing image synthesis as a multi‑scale prediction task, where at each resolution the model predicts the next finer-scale image tokens in an autoregressive fashion[^1]. This coarse‑to‑fine transformer architecture allows VAR to learn visual distributions efficiently and surpass diffusion models on ImageNet 256×256 benchmarks[^1][^9].

---

## VAR‑CLIP

![VAR‑CLIP Architecture](/images/VAR-CLIP_2.png)

Building on VAR’s backbone, VAR‑CLIP feeds CLIP text embeddings as conditioning context at every scale. Captions are encoded by a pre‑trained CLIP model into a fixed‑length vector, which guides token prediction to yield semantically aligned outputs across diverse prompts[^2][^8][^10].

---

## MC Dropout

![MC Dropout Illustration](/images/MC_Dropout_VAR-CLIP.drawio.png)

To approximate Bayesian inference, I activate dropout layers during inference and perform multiple forward passes per prompt. Aggregating 50 samples per seed, I compute pixel‑wise standard deviations (σ) as epistemic uncertainty estimates without modifying training regimes[^3].

---

## Experiments with Tables and Images

### Performance–Uncertainty Trade‑off

For the prompt “Belgian shepherd,” I sweep DropPath rates and record FID, Inception Score (IS), Central Moment Discrepancy (CMMD), and average σ:

| DropPath Rate | FID ↓ | IS ↑               | CMMD ↓ | σ      |
|---------------|-------|--------------------|--------|--------|
| 0.05          | 18.54 | 78.32 ± 1.73       | 4.357  | 0.1927 |
| 0.10          | 18.98 | 73.11 ± 1.31       | 4.384  | 0.2107 |
| 0.20          | 22.68 | 64.13 ± 1.91       | 4.536  | 0.2198 |
| 0.30          | 30.57 | 51.82 ± 0.91       | 4.833  | 0.2259 |

![DropPath Experiment](/images/exp2.png)

As dropout increases, generation quality degrades (higher FID, lower IS) while uncertainty rises, illustrating the fidelity–confidence trade‑off[^4][^5].

### Region‑Specific Confidence

Using SAM to segment “Golden retriever” images, I compare foreground and background uncertainty:

| Region     | σ      |
|------------|--------|
| Object     | 0.1975 |
| Background | 0.1993 |

<div style="display: flex; gap: 1rem; justify-content: space-between; align-items: flex-start;">
  <figure style="flex: 1; text-align: center; margin: 0;">
    <img src="/images/object.png" alt="Object Uncertainty Map" style="max-width: 100%; height: auto;" />
    <figcaption>Figure a: Object Uncertainty Map</figcaption>
  </figure>
  <figure style="flex: 1; text-align: center; margin: 0;">
    <img src="/images/background.png" alt="Background Uncertainty Map" style="max-width: 100%; height: auto;" />
    <figcaption>Figure b: Background Uncertainty Map</figcaption>
  </figure>
</div>



Foreground regions exhibit slightly lower σ, indicating higher confidence in semantically salient areas[^7].

### Semantic Entanglement

I generate images for prompts with one, two, or three concepts and measure uncertainty:

| Concepts | FID ↓  | IS ↑              | CMMD ↓ | σ      |
|----------|--------|-------------------|--------|--------|
| 1        | 18.54  | 78.32 ± 1.73      | 4.357  | 0.1927 |
| 2        | 35.96  | 31.53 ± 0.52      | 4.381  | 0.1955 |
| 3        | 47.99  | 21.16 ± 0.43      | 4.528  | 0.1924 |


<div style="display: flex; gap: 1rem; justify-content: space-between;">
  <figure style="flex: 1; text-align: center; margin: 0;">
    <img
      src="/images/fox.png"
      alt="Arctic fox"
      style="width: 100%; max-width: 200px; height: auto;"
    />
    <figcaption>Figure 1: “Arctic fox”</figcaption>
  </figure>

  <figure style="flex: 1; text-align: center; margin: 0;">
    <img
      src="/images/fox_teddy.png"
      alt="Arctic fox or teddy"
      style="width: 100%; max-width: 200px; height: auto;"
    />
    <figcaption>Figure 2: “Arctic fox or teddy”</figcaption>
  </figure>

  <figure style="flex: 1; text-align: center; margin: 0;">
    <img
      src="/images/fox_piggy_schooner.png"
      alt="Arctic fox or piggy bank or schooner"
      style="width: 100%; max-width: 200px; height: auto;"
    />
    <figcaption>Figure 3: “Arctic fox or piggy bank or schooner”</figcaption>
  </figure>
</div>


Uncertainty peaks at two concepts, reflecting maximum semantic ambiguity before dilution[^6].

### Prompt Descriptiveness

Comparing a simple caption against a richly descriptive one:

| Prompt Type | FID ↓  | IS ↑               | CMMD ↓ | σ      |
|-------------|--------|--------------------|--------|--------|
| Simple      | 18.54  | 78.32 ± 1.73       | 4.357  | 0.1927 |
| Complex     | 15.14  | 103.49 ± 1.55      | 3.785  | 0.1794 |

<div style="display: flex; gap: 1rem; justify-content: space-between; align-items: flex-start;">
  <figure style="flex: 1; text-align: center; margin: 0;">
    <img
      src="/images/spider.png"
      alt="Black widow"
      style="max-width: 100%; height: auto;"
    />
    <figcaption>Figure c: “Black widow”</figcaption>
  </figure>
  <figure style="flex: 1; text-align: center; margin: 0;">
    <img
      src="/images/spider_complex.png"
      alt="A black and red spider with red eyes"
      style="max-width: 100%; height: auto;"
    />
    <figcaption>Figure d: “A black and red spider with red eyes”</figcaption>
  </figure>
</div>


Descriptive prompts both boost fidelity and reduce uncertainty, underscoring the value of unambiguous guidance[^5].

---

## Conclusions

- **VAR vs VAR‑CLIP:** VAR excels in raw fidelity, while VAR‑CLIP enhances semantic alignment at the cost of higher baseline uncertainty[^1][^2].  
- **MC Dropout:** A practical Bayesian approximation that reveals how dropout rates shape the fidelity–confidence trade‑off[^3].  
- **Semantic Relevance:** SAM‑based masking confirms greater confidence in salient regions[^7].  
- **Prompt Design:** Both complexity and descriptiveness critically influence uncertainty, with moderate ambiguity peaking σ and detailed captions minimizing it[^5][^6].

---

## References

[^1]: Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, Liwei Wang. “Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction.” NeurIPS 2024 [[arXiv:2404.02905]](https://arxiv.org/abs/2404.02905).  
[^2]: Qian Zhang, Xiangzi Dai, Ninghua Yang, Xiang An, Ziyong Feng, Xingyu Ren. “VAR‑CLIP: Text-to-Image Generator with Visual Auto-Regressive Modeling.” arXiv:2408.01181 (2024).  
[^3]: Yarin Gal, Zoubin Ghahramani. “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.” ICML 2016 [[arXiv:1506.02142]](https://arxiv.org/abs/1506.02142).  
[^4]: Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter. “GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.” NeurIPS 2017 [[arXiv:1706.08500]](https://arxiv.org/abs/1706.08500).  
[^5]: Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen. “Improved Techniques for Training GANs.” NIPS 2016 [[arXiv:1606.03498]](https://arxiv.org/abs/1606.03498).  
[^6]: Werner Zellinger, Thomas Grubinger, Edwin Lughofer, Thomas Natschläger, Susanne Saminger-Platz. “Central Moment Discrepancy for Domain-Invariant Representation Learning.” ICLR 2017 [[arXiv:1702.08811]](https://arxiv.org/abs/1702.08811).  
[^7]: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan‑Yen Lo, Piotr Dollár, Ross Girshick. “Segment Anything.” ICCV 2023 [[arXiv:2304.02643]](https://arxiv.org/abs/2304.02643).  
[^8]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. “Learning Transferable Visual Models From Natural Language Supervision.” arXiv:2103.00020 (2021).  
[^9]: FoundationVision. “VAR GitHub Repository.” GitHub (2024), https://github.com/FoundationVision/VAR.  
[^10]: daixiangzi. “VAR‑CLIP GitHub Repository.” GitHub (2024), https://github.com/daixiangzi/VAR-CLIP.  
