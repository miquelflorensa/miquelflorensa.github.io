# My Style Guide

This document captures my personal style, standards, and conventions so that Claude (or future-me) can replicate my work consistently. It is organized by category and will grow over time.

---

## 1. Figures

### 1.1 General Philosophy

- Figures must be **publication-ready** from the script. No manual post-processing.
- Every aesthetic choice (sizes, fonts, colors, spacing) is defined as a **named constant at the top** of the script in a dedicated configuration section, never buried inside plotting logic.
- Scripts are **modular**: separate functions for data computation, individual plot elements (Gaussians, error bars, boxes), and the overall figure assembly.
- **Always save both PDF (vector) and PNG (raster)** for every figure. PDF goes in the paper; PNG is for presentations and quick previews.
- Close figures after saving (`plt.close(fig)`) to avoid memory leaks in batch runs.

### 1.2 Matplotlib & LaTeX Setup

Always at the top of every script:

```python
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'
```

This ensures Times font throughout, matching the paper body. LaTeX rendering is non-negotiable for any text that contains math.

For scripts with fine-grained size control, also set:

```python
plt.rc('font', family='serif', size=8)
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('legend', fontsize=6)
```

When using seaborn for exploration/comparison plots (not final paper figures):

```python
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
```

### 1.3 Figure Dimensions

I target **Physical Review D (PRD)** column widths:

```python
pt = 1./72.27
jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt}}
```

| Context | Width | Typical height |
|---|---|---|
| Single-column (most figures) | `3.25"` | `1.8"` -- `4.2"` depending on content |
| Two-column (comparison/overview) | `6.75"` -- `7.0"` (or `jour_sizes["PRD"]["twocol"]`) | golden ratio or `2.4"` -- `5.0"` |

For golden-ratio figures:

```python
golden = (1 + 5 ** 0.5) / 2
figsize = (my_width, my_width / golden)
```

### 1.4 Font Sizes (Hierarchy)

| Element | Size | Notes |
|---|---|---|
| Subplot titles | `8`--`9` pt | Always `fontweight='bold'` |
| Axis labels | `8` pt | |
| Tick labels | `7` pt | |
| Legend | `6`--`8` pt | Smaller in compact figures |
| Annotations/text boxes | `7`--`11` pt | Context-dependent |
| Figure-level side labels | `11`--`12` pt | For flow diagrams |

### 1.5 Color Palette

#### Primary categorical (3-class):
| Role | Hex | Name |
|---|---|---|
| Class 0 / Cat | `#2E86AB` | Blue |
| Class 1 / Dog | `#E94F37` | Red |
| Class 2 / Bird | `#F39C12` | Orange |

#### Semantic colors:
| Role | Hex |
|---|---|
| Arrow / connector (neutral) | `#2C3E50` |
| Forward pass | `#2E7D32` (green) |
| Backward pass | `#C62828` (red) |
| Sum / total | `#0B5345` (deep dark green) |

#### Box / region backgrounds:
| Role | Hex |
|---|---|
| BNN | `#B7E6B5` (light green) |
| Prior | `#FFE5B4` (light orange) |
| Remax box | `#AED6F1` (light blue) |
| Update / observation | `#F9E79F` (light yellow) |
| Posterior | `#CAFFBF` (light green) |

#### Method comparison (multi-method plots):
| Method | Hex |
|---|---|
| LL-Softmax | `#4A90E2` (blue) |
| MM-Softmax | `#B8E986` (green) |
| MM-Remax | `#9B59B6` (purple) |
| MM-LaplaceCDF | `#E74C3C` (red) |
| Probit/NormCDF | `#F5A623` (orange) |

#### Metric-specific colors (training curves):
| Metric | Hex | Marker |
|---|---|---|
| mu (mean) | `#4A90E2` | `'o'` circle |
| sigma (std) | `#E24A4A` | `'s'` square |
| rho (correlation) | `#50C878` | `'^'` triangle |

### 1.6 Spine & Grid Conventions

**Always remove top and right spines:**

```python
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

For distribution plots (PDF curves), also remove the left spine and y-ticks:

```python
ax.spines['left'].set_visible(False)
ax.set_yticks([])
```

**Gridlines:** Use sparingly. When present: `ax.grid(True, alpha=0.3)`. For clean paper figures, prefer no grid. For exploratory/comparison plots, subtle grid is acceptable.

### 1.7 Error Bars (Horizontal)

My standard error bar style for showing mean +/- std of probabilities or logits:

```python
# Horizontal error bar line
ax.errorbar(mean, y_pos, xerr=std, fmt='none', ecolor=color, elinewidth=1.0, capsize=2)

# Vertical caps at endpoints (manual, for extra clarity)
ax.plot([mean - std]*2, [y_pos - 0.2, y_pos + 0.2], color, lw=1.0)
ax.plot([mean + std]*2, [y_pos - 0.2, y_pos + 0.2], color, lw=1.0)

# Square marker at the mean
ax.scatter(mean, y_pos, color=color, marker='s', s=10, zorder=3, edgecolor='k', linewidth=0.3)
```

Key choices:
- **Square markers** (`'s'`) for means, not circles.
- Black edge on markers (`edgecolor='k'`, `linewidth=0.3`--`0.5`).
- `zorder=3` to keep markers on top.
- `invert_yaxis()` so class 0 is at top.

### 1.8 Gaussian Distributions (PDF Curves)

```python
x = np.linspace(x_min, x_max, 500)  # 200-500 points
y = norm.pdf(x, mu, sigma)

ax.plot(x, y, color=color, linewidth=1.0)          # Analytical (solid)
ax.fill_between(x, y, alpha=0.2-0.3, color=color)  # Shaded area

# Monte Carlo comparison (dashed)
kde = gaussian_kde(mc_samples)
ax.plot(x, kde(x), color=color, lw=1.0, ls='--', alpha=0.8)
```

Conventions:
- **Solid lines** = analytical / moment-matching results.
- **Dashed lines** = Monte Carlo estimates.
- Fill alpha: `0.2`--`0.3` (lighter for overlapping distributions).
- When used inline (inset axes): `linewidth=1.8`, `alpha=0.3`.

### 1.9 Legend

```python
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color='black', lw=1.0, label='Moment-Matching'),
    Line2D([0],[0], color='black', lw=1.0, ls='--', label='Monte Carlo')
]
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.8, fontsize=6, handlelength=1.5)
```

- Small font (6--8pt).
- Semi-transparent frame (`framealpha=0.8`--`0.9`).
- Position: `upper right` or `upper left` with `bbox_to_anchor` for fine control.

### 1.10 Subplot Titles

Always use LaTeX bold numbering:

```python
ax.set_title(r'\textbf{(a)} Description', fontsize=9, pad=3)
# or
ax.set_title(r'\textbf{1. Input} $Z$', fontsize=9, pad=3)
```

### 1.11 Heatmaps / Imshow

```python
im = ax.imshow(data, origin='lower', aspect='auto',
               extent=(x_min, x_max, y_min, y_max),
               cmap='viridis', vmin=vmin, vmax=vmax)
```

| Data type | Colormap |
|---|---|
| Expected value / probability | `viridis` |
| Absolute error | `inferno` |
| MAE (log scale) | `plasma` |
| General heatmap | `YlOrRd` |

Colorbars:

```python
fig.colorbar(im, ax=[ax0, ax1],         # Shared across panels
             fraction=0.05, pad=0.04,
             label=r'$E[A_{\mathrm{dom}}]$',
             location='bottom',
             shrink=1.0, aspect=15-30)
```

### 1.12 Box Plots

```python
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                showfliers=False, widths=0.6)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
```

- `showfliers=False` -- I never show outliers.
- `patch_artist=True` to fill boxes with color.
- Box alpha: `0.7`.

### 1.13 Arrows & Flow Diagrams

```python
arrow = FancyArrowPatch(start, end,
                        transform=fig.transFigure,
                        arrowstyle='-|>',
                        mutation_scale=20-25,
                        linewidth=2.5,
                        color=color,
                        connectionstyle='arc3,rad=0')
fig.patches.append(arrow)
```

Arrow labels with boxed text:

```python
fig.text(mid_x, mid_y, label, fontsize=10, ha='center', va='center',
         fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor=color, linewidth=2))
```

Rounded boxes for components:

```python
box = FancyBboxPatch(xy, width, height,
                     boxstyle="round,pad=0.05",
                     facecolor=color, edgecolor='black',
                     linewidth=2)
```

### 1.14 GridSpec Layouts

```python
gs = fig.add_gridspec(rows, cols,
                      height_ratios=[...],
                      width_ratios=[...],
                      hspace=0.6, wspace=0.0-0.4)
```

| Layout | Use case |
|---|---|
| 4x2, `height_ratios=[0.7, 1.2, 0.7, 1.2]` | Error bars + distributions (alternating compact/full rows) |
| 5x3 | Flow diagrams |
| 1x2, 1x3 | Side-by-side method comparisons |
| 2x3 | Multi-metric, multi-condition grids |

Margins for flow diagrams:

```python
LEFT=0.08, RIGHT=0.92, TOP=0.95, BOTTOM=0.08
```

Use `constrained_layout=True` for automatic spacing in simpler figures. Use explicit `GridSpec` for complex layouts.

### 1.15 Training Curve Plots

```python
ax.plot(epochs, mean, color=color, linewidth=2,
        marker=marker, markersize=4, markevery=5, label=label)

ax.fill_between(epochs, mean - std, mean + std,
                color=color, alpha=0.2, edgecolor='none')
```

- Error bands at +/-1 std with `alpha=0.2`.
- Markers every 5 epochs, not every point.
- Grid: `alpha=0.3`.
- `xlim` set to exact epoch range.

### 1.16 Saving Figures

```python
# Standard save (always both formats)
fig.savefig(output_path_pdf, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig(output_path_png, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
```

- **Always `dpi=300`**.
- **Always `bbox_inches='tight'`** to crop whitespace.
- White background, no edge.
- Save to a `figures/` directory (created with `mkdir(parents=True, exist_ok=True)`).

### 1.17 Script Structure

Every figure script follows this skeleton:

```python
"""
Docstring explaining what the figure shows and its layout.
"""

import numpy as np
import matplotlib.pyplot as plt
# ... other imports ...

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# LaTeX setup
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=8)
# ... etc ...

# Figure dimensions
FIG_WIDTH = 3.25
FIG_HEIGHT = 4.2

# Colors
COLORS = ['#2E86AB', '#E94F37', '#F39C12']

# Data / parameters
MU_Z = np.array([...])

# ============================================================================
# END OF CONFIGURATION
# ============================================================================


def compute_something(...):
    """Data computation, separate from plotting."""
    ...

def draw_element(ax, ...):
    """Draw a single reusable plot element."""
    ...

def create_figure():
    """Assemble the complete figure."""
    # 1. Compute data
    # 2. Create figure and gridspec
    # 3. Plot each panel
    # 4. Save
    ...

if __name__ == '__main__':
    create_figure()
```

---

## 2. Writing

### 2.1 General Philosophy

- Writing must be **dense and precise**; every sentence carries information. No padding, no filler, no sentences that exist only to sound nice.
- I write for a **technical audience** that understands the domain. I do not over-explain well-known concepts; I state what I need from them and move on.
- The tone is **confident and direct**, but never arrogant. I make strong claims when backed by math or experiments and openly acknowledge limitations when they exist.
- I use **first person plural** ("we") throughout, even in single-author work, this is my convention for academic writing.
- Precision over accessibility: I prefer the exact technical term over a simplified paraphrase.

### 2.2 Vocabulary: Words and Phrases I Actually Use

These are recurring words in my writing. An LLM helping me should draw from this vocabulary, not from generic academic synonyms:

| Word/Phrase | How I use it |
|---|---|
| `tractable` / `intractable` | Core to my work; I use these very frequently and precisely for analytical computability |
| `analytically` | As an adverb: "analytically compute," "analytically tractable," "analytically update" |
| `formulation` | For a mathematical setup or framework, not "approach" or "methodology" |
| `instantiation` | For a specific implementation of a general method (e.g., "two instantiations of this method") |
| `regime` | For parameter regions: "high-variance regimes," "low-data regimes" |
| `practitioners` | For people who use the methods, not "users" or "researchers" |
| `end-to-end` | For processes that work from input to output without manual steps |
| `closed-form` | For exact analytical solutions |
| `propagate` | For moving uncertainty or information through layers: "propagate uncertainty," "propagate posterior information" |
| `conjugacy` | In the Bayesian sense; I use it naturally |
| `decompose` / `decomposition` | For breaking problems into parts |
| `leveraging` | I use it occasionally and naturally (e.g., "leveraging the conditional independence"), but not as a filler verb |
| `employ` | I use it as a neutral synonym for "use" in formal contexts: "TAGI employs the GMA" |
| `modulate` | For describing how one quantity affects another: "correctly modulates the expected value" |
| `denoted` / `denoted as` | For introducing notation: "denoted $\bm{Z} \sim \mathcal{N}(...)$" |
| `reside on` | For mathematical objects on spaces: "the true output probabilities reside on the simplex" |
| `incur` | For costs or errors: "incur significant precision loss" |
| `render` | Causal: "the exponential formulation renders this modulation highly inaccurate" |
| `ad-hoc` | For informal workarounds: "without requiring ad-hoc stabilization heuristics" |
| `strictly` | As an intensifier for mathematical precision: "strictly positive," "strictly invariant," "strictly prior to divergence" |
| `negligible` | For small errors: "maintaining a negligible error" |
| `non-negotiable` | Rare, for absolute requirements |

### 2.3 Vocabulary: Words and Phrases I NEVER Use (That LLMs Love)

This is critical. If you are writing for me, **never use any of these**:

| Banned word/phrase | Why |
|---|---|
| `delve` / `delve into` | Classic LLM filler. I never use it. |
| `crucial` / `crucially` | Overused LLM intensifier. I use "essential," "required," or "strict" instead. |
| `it is worth noting that` | Padding. If it's worth noting, just say it. |
| `importantly` / `more importantly` | I don't front-load sentences with importance flags. The structure implies importance. |
| `notably` (at sentence start) | I occasionally use "notably" mid-sentence to flag a specific observation (e.g., "Notably, in this approximation..."), but never as a generic sentence opener to flag importance. |
| `it should be noted that` | Pure padding. Delete and keep the content. |
| `in summary` / `to summarize` | I do not summarize at the end of sections. The conclusion stands on its own. |
| `a myriad of` | Pretentious. I say "a wide range of" or "various" if needed. |
| `in the realm of` | Flowery. I say "in" or "for." |
| `shed light on` | Metaphor I avoid. I say "clarify," "reveal," or just describe the finding directly. |
| `plays a crucial role` / `plays a key role` | Double offender (vague verb + LLM intensifier). I state the function directly. |
| `this is particularly useful because` | Explaining motivation mid-stream. I state the motivation up front or let it be obvious. |
| `it is important to emphasize` | If it's important, the sentence structure and position do the emphasizing. |
| `pave the way` | Cliché. I say "enables" or "lays the foundations for." |
| `landscape` (as metaphor) | "The ML landscape". I don't use this. |
| `harness` / `harnessing` | "Harnessing the power of". No. I use "leveraging" sparingly or just "using." |
| `unveil` / `uncover` | Dramatic verbs I avoid. I "introduce," "present," or "propose." |
| `novel` (overused) | I use "novel" at most once per paper, for the genuinely new thing. Never as a generic adjective. |
| `straightforward` | I avoid it. If it's simple, the math shows it. |
| `paradigm` | Only if referring to an actual paradigm (e.g., "inference paradigms" as a category). Never as a buzzword. |
| `robust` (as filler) | I DO use "robust" but only when I mean it specifically; stable under perturbation, doesn't diverge. Never as a vague positive adjective. |
| `explore` | I don't "explore" things. I "analyze," "evaluate," "assess," or "investigate." |
| `exciting` / `promising` / `remarkable` / `groundbreaking` | Hype words. I never use motivational language about results. |
| `In this section, we will explore/discuss...` | Roadmap filler at section starts. I jump straight into content. |
| `Let us now turn to...` | Artificial transition. I use a section heading or a direct statement. |
| `We now proceed to...` | Same, never. |
| `The key takeaway is...` | I don't editorialize. The reader draws their own takeaways. |
| Em-dashes (`—` or ` -- `) | I never use em-dashes. LLMs overuse them. I restructure with semicolons, commas, or parentheses. |
| Colon before an equation | Equations flow from the sentence. Never "...is given by:" or "...defined as:" with a colon. |

### 2.4 Sentence Structure

- **Topic sentence → technical detail → implication.** This is my paragraph skeleton. The first sentence states what is happening or what I claim. The middle sentences give the math or evidence. The last sentence states the consequence.
- **I front-load information.** The subject and verb come early. I don't bury the main point behind subordinate clauses.
- **Sentences are medium-length and dense**, not short and choppy, not long and winding. A typical sentence has one main clause and at most one subordinate clause.
- **I use semicolons correctly and often** to join closely related independent clauses rather than starting a new sentence. Example: "the exponential nature of Softmax amplifies logit variance, leading to numerical overflow in the variance and covariance computations essential for the backward pass."
- **Parenthetical precision:** I use parentheses to add clarifying specifics inline: "(i.e., $\bm{Z}^{(j-1)} \perp \bm{Z}^{(j+1)} | \bm{z}^{(j)}$)" or "(see Appendix~\ref{...})".
- **Appositive definitions:** I define terms in apposition right when I introduce them: "a zero-mean Gaussian noise term, $\bm{V} \sim \mathcal{N}(\bm{0}, \bm{\Sigma}_{\bm{V}})$"; not in a separate sentence.

### 2.5 Transitions and Connectors

Transitions I actually use (draw from these, not from generic LLM connectors):

| Transition | When I use it |
|---|---|
| `Consequently,` | Direct causal result. My most common transition |
| `However,` | Contrast or limitation |
| `To address...` | Introducing a solution after stating a problem |
| `Because...` | Direct causal at sentence start. I do this and it's fine |
| `While...` | Concessive contrast: "While X works, Y fails" |
| `Furthermore,` | Rare, for genuine additions |
| `Notwithstanding...` | Very rare, for limitations: "Notwithstanding these limitations,..." |
| `In contrast,` | Direct method-to-method comparison |
| `As shown in...` | Referencing a figure or equation |

Transitions I **never use** as sentence openers:
- `Moreover,`. I use `Furthermore,` if anything, and rarely.
- `Additionally,`. I either restructure or use "also" mid-sentence.
- `Thus,`. I prefer `Consequently,` or restructure the sentence.
- `Hence,`. Same; I avoid it.
- `Indeed,`. LLM filler. Never.
- `Interestingly,`. Editorializing. Never.
- `Clearly,`. If it were clear, I wouldn't need to say so.
- `Obviously,`. Same problem.
- `Of note,`. Padding.
- `Taken together,`. Padding before a summary.

### 2.6 How I Introduce Equations

Equations are part of the sentence grammar. They flow naturally from the text. They are not standalone objects that get introduced with fanfare or preceded by colons.

**Numbering convention:** Equations are **unnumbered by default** (use `align*`, `equation*`). Only number an equation if it is referenced later in the text (e.g., via `\ref`) or if it represents a key result or conclusion of a derivation. Most intermediate steps stay unnumbered.

**Patterns I use:**
- "...is given by" → equation (no colon, the sentence flows into the math)
- "...is defined as" → equation
- "...such that" → condition/constraint
- "...with moments" → followed by the equations
- "...denoted $X \sim \mathcal{N}(...)$"
- "where $\alpha_i = \mu_{Z_i}/\sigma_{Z_i}$, and $\phi, \Phi$ denote...". The "where" line after an equation defines all new symbols.
- "Substituting these into...". For chaining derivations.

**Patterns I never use:**
- "The following equation shows...". Too indirect.
- "We can write this as...". Padding.
- "It can be easily shown that...". If it's easy, show it or skip it.
- "After some algebra, we get...". Either show the algebra (in appendix) or just state the result.
- Any use of a colon (`:`) right before an equation. The sentence must flow grammatically into the math without punctuation interruption.

### 2.7 How I Introduce Methods and Concepts

- I **define before I name**: I explain what the thing does, then give it a name. Not the other way around. Example: "a novel function that replaces the exponential with a rectified linear unit to enhance numerical stability in high-variance regimes", the definition, followed by calling it MM-Remax.
- I use **bold** or `\textbf{}` for the first occurrence of a named method, then plain text after.
- I **state conditions and requirements as numbered lists** when they are formal. Example: "a tractable classification layer must satisfy two strict analytical conditions: 1. Forward Tractability... 2. Backward Tractability..."
- I **name conditions in bold** and then explain them: "\textbf{Forward Tractability:} The first two moments..."

### 2.8 How I Handle Related Work

- I organize related work **chronologically within conceptual groups**, not as a flat list of papers.
- I **position my work relative to others explicitly**: I say what others did, what gap remains, and where I fit.
- I give **specific technical reasons** why prior work is insufficient, not vague criticism. Example: "they focus strictly on computing the prior predictive of the categorical probabilities. Consequently, they do not allow computing the cross-covariance term."
- I never say "X et al. did great work but...". I state what they did and what's missing without performative praise.
- I **cite concretely**: I point to specific results or properties, not to entire papers vaguely.

### 2.9 How I Write Experimental Sections

- I split experiments into **Verification** (controlled/synthetic, checking correctness) and **Validation** (real benchmarks, checking utility). This is a deliberate distinction.
- I use bold headers within experiments to segment: **Experimental Setup.**, **Training Stability and Convergence.**, **Generalization Performance.**, **Qualitative Comparison.**, **Quantitative Error Analysis.**
- For setup paragraphs: architecture → hyperparameters → evaluation protocol → reporting (seeds, averaging). Dense, no fluff.
- For results: I **state the finding first, then point to the figure/table**. Not: "Figure 3 shows..." but "MM-Remax exhibits robust convergence... Figure 3 illustrates these training dynamics."
- I report results as "Mean ± Std. over N seeds" and always state the number of seeds.
- Failure modes are reported explicitly and without softening: "training inevitably diverges, eventually resulting in numerical collapse, i.e., NaN outputs."

### 2.10 How I Write Abstracts

My abstract follows a strict 5-part structure in a **single dense paragraph**:

1. **Context/Problem** (1--2 sentences): What the field needs and why it's hard.
2. **Gap** (1 sentence): What's missing.
3. **Proposal** (2--3 sentences): What I do, stated concretely.
4. **Methods** (1--2 sentences): The specific things I propose, by name.
5. **Results** (1--2 sentences): Where I tested, what I found.

No motivational opening ("Deep learning has revolutionized..."). I go straight to the technical problem.

### 2.11 How I Write Introductions

1. **Opening paragraph:** State the challenge and define the problem precisely. Introduce formal notation if needed early.
2. **Figure early, but never before its reference:** I place a key figure (often Figure 1) in the introduction to give the reader an immediate visual anchor, but the figure always appears after its first reference in the text. A figure must never float above the paragraph that first cites it.
3. **Motivation paragraph:** Why existing approaches are insufficient, with citations.
4. **Formal conditions:** If the method must satisfy specific properties, I list them as a numbered list with bold names.
5. **Proposed methods:** I describe what I propose, organized by approach class.
6. **Contributions:** Always a bulleted/numbered list, starting with "Our contributions are as follows:"; each item is one sentence, concrete and verifiable.

### 2.12 How I Write Conclusions

- **Short**; typically one paragraph for the summary, one for limitations, one for future work.
- I **restate the main result** in one sentence, not a recap of the whole paper.
- I **explicitly list limitations** ("We identify two primary limitations. First,... Second,..."). I am honest and specific, not vague.
- Future work is stated as concrete directions, not generic "future work could explore...". I say exactly what the next step enables.
- The last sentence is **forward-looking and specific**, often tying to a broader application: "this work lays the foundations for the development of analytically tractable attention mechanisms."

### 2.13 Figure Captions and Placement

- **A figure must never appear before its first reference in the text.** If I reference Figure 3 in paragraph X, Figure 3 cannot float above paragraph X. Use `[t]`, `[H]`, or placement hints to enforce this.
- Captions are **self-contained**: a reader should understand the figure from the caption alone, without reading the main text.
- I use **bold labels** within captions to segment multi-panel figures: "\textbf{(1)} input Gaussian logits, through \textbf{(2)} the element-wise exponential..."
- I state **what the figure shows**, not what the reader should feel about it.
- I include method details in captions when needed: "(MC sampling, $\mathtt{N}=10^6$)."
- For comparison figures: "Solid lines represent... dashed lines represent..."
- I keep captions to 2--4 sentences.

### 2.14 Table Conventions

- I use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`); no vertical lines.
- Footnotes below tables use symbols ($\dagger$), not numbers.
- I include "Mean $\pm$ Std." in the caption and state "over N seeds."
- Failed methods are marked explicitly with a footnote explaining the failure mode.

### 2.15 Notation Conventions

Two independent axes define the notation:
- **Bold** means vector: $\bm{z}$ is a vector, $z_i$ is a scalar component.
- **Uppercase** means random variable: $Z$ is a random variable, $z$ is a realization (observed value).

These combine naturally:
- $\bm{Z}$: random vector.
- $\bm{z}$: observed vector (realization).
- $Z_i$: scalar random variable (component $i$).
- $z_i$: observed scalar.

Other notation:
- Expected values: $\mu_{Z_i}$ or $\bm{\mu}_{\bm{Z}}$.
- Variances: $\sigma_{Z_i}^2$ or $\bm{\sigma}_{\bm{Z}}^2$.
- Covariance matrices: $\bm{\Sigma}_{\bm{Z}}$.
- I use `\text{Cov}(X, Y)` for covariance, `\text{Var}[X]` for variance, and $\mathbb{E}[\cdot]$ for expectation.
- Calligraphic $\mathcal{N}$ for Gaussian distributions.
- Layer indices as superscripts in parentheses: $\bm{Z}^{(j)}$, $\bm{Z}^{(\mathtt{O})}$ for output.
- Typewriter font for constants/counts: $\mathtt{K}$ (number of classes), $\mathtt{N}$ (number of samples).

### 2.16 Punctuation and Formatting Micro-rules

- **i.e.,** and **e.g.,** are always followed by a comma: "i.e., the input uncertainty does not influence..."
- **Never use em-dashes** (`—` or `--`). English is not my first language and I do not use them. Restructure using semicolons, commas, parentheses, or separate sentences. LLMs love em-dashes; I do not.
- **Colons** introduce lists and definitions, but **never equations**. Equations flow as part of the sentence grammar (see §2.6).
- No **Oxford comma** inconsistency; I use it consistently.
- Section references use `§` (e.g., `\S\ref{sec:verification}`), not "Section."
- I use `\vspace{-1em}` and `\vspace{-2em}` to tighten vertical spacing in the final layout.

### 2.17 Things I Do That LLMs Typically Skip

- **I acknowledge failure modes explicitly.** I don't hide them or soften them. If a method diverges, I say "inevitably diverges" and "resulting in NaN outputs."
- **I compare against my own methods honestly.** If MM-Remax gets a lower accuracy than Logit Regression on some benchmark, I report it. I don't cherry-pick.
- **I give equations agency in sentences.** "Substituting these into the moment-matching process produces the output moments"; the math is the subject.
- **I use passive voice intentionally and sparingly** for methodological descriptions where the actor is irrelevant: "Weights and biases are initialized using a default gain factor of 0.1."
- **I define acronyms on first use** and use them consistently after: "Tractable Approximate Gaussian Inference (TAGI)" → then "TAGI" everywhere.
- **I use "rather than"** for contrasts instead of "instead of" at sentence level.
- **I compress related-work citations** with semicolons: "foundational approaches \citep{A, B, C} have been expanded upon by modern methods."

---

## 3. Code

*To be added.*