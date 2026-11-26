# Fast Prediction Framework (Final)

> Stage: **Final** (~20 pages) · Status: **Stable v1.4** · Last updated: 2026-11-26
> Language: [中文](../../最终版/快速预测框架.md) · English

## 0. Notation

| Symbol | Meaning |
| --- | --- |
| $\mathcal{V}$ | Full vocabulary, $N = \lvert\mathcal{V}\rvert$, typically $10^4 \sim 10^5$ |
| $x$ | Input sentence, modeled as a **word set** (unordered, existence only) |
| $y$ | Output answer, an ordered word sequence |
| $C$ | Compressed vocabulary, $C \subseteq \mathcal{V}$, $\lvert C\rvert = k \ll N$ |
| $T(y)$ | Set of distinct words used by the answer |
| Coverage | Fraction of $T(y)$ contained in $C$ |
| Closure $\mathcal{G}$ | Fixed set of reserved words (punctuation, EOS, etc.) |

## 1. Abstract

A standard language model computes a softmax over the full vocabulary $\mathcal{V}$ at every step and searches over it, so both compute and memory grow linearly with $\lvert\mathcal{V}\rvert$. But a real answer uses only a few dozen distinct words. The **Fast Prediction Framework** (FPF) explicitly splits "predict the next word" into two modules:

1. **Vocabulary Distribution Prediction**: takes an unordered word set, outputs per-word existence probabilities over the whole vocabulary, and selects the compressed vocabulary $C$;
2. **Word-String Generation**: autoregressively composes the answer inside $C$ only; every output word must come from $C$.

The factorization is $p(y \mid x) \approx p(C \mid x)\, p(y \mid C)$: the compressed vocabulary $C$ is an unordered, existence-only summary and necessarily discards word order and part of the context. It is a **deliberate information bottleneck** whose cost is measured by coverage and triple consistency.

The whole framework scales linearly with sequence length: the input side performs a single linear scan and never uses Transformer-style $O(n^2)$ global attention (2.4). Each module has several implementation options (Chapters 6 and 7), and optimization targets triple consistency (Chapter 8). Training adds input noise injection and online learning (Chapter 9), and the Li-MCTS graph grows adaptively with use (7.2.5).

## 2. Problem Definition

### 2.1 Standard Objective

$$
p(y \mid x) = \prod_{t=1}^{L} p(y_t \mid x, y_{<t}), \qquad y_t \in \mathcal{V}
$$

Each step requires:

1. the output layer to map a $d$-dimensional hidden state to $N$ logits ($O(N d)$);
2. a softmax and sampling/comparison over $N$ candidates;
3. beam search to compare $B \cdot N$ candidates per step.

### 2.2 Cost Structure

| Term | Standard LM | This framework |
| --- | --- | --- |
| Output-layer compute | $O(L \cdot N \cdot d)$ | $O(N \cdot d) + O(L \cdot k \cdot d)$ |
| Output logits memory | $O(L \cdot N)$ | $O(N) + O(L \cdot k)$ |
| Search comparisons per step | $O(B \cdot N)$ | $O(B \cdot k)$ |

With $N = 10^5$, $k = 512$, $L = 32$, and $d = 256$, output-layer multiply-adds drop from about $8.2 \times 10^8$ to about $3.0 \times 10^7$, roughly **27×**. With $k = 256$ it is about 30×.

This counts the output layer only; the $O(n^2)$ input-side cost of global attention is separate and forbidden by the design principle in 2.4.

### 2.3 Key Observation

The support $T(y)$ of the answer is small and largely determined by the input:

- in dialogue, answer words mostly come from words already present in the input (the copy channel);
- in rewriting/summarization, content words are determined by the semantics of the input;
- function words and punctuation are almost fixed (the closure channel).

So "which words may appear" can be **computed once**, while "how to arrange them" is the real language-modeling problem.

### 2.4 Design Principle: Linear Complexity

The framework's compute must grow linearly with sequence length $n$, not quadratically like Transformer self-attention.

- **Module 1**: a single linear scan over the input — per-token transform (token MLP or linear attention) plus masked pooling, $O(n d^2)$; no pairwise attention inside the input;
- **Module 2**: works only over the small vocabulary $C$ and the short answer length $L$; it never re-reads the long input; context enters as an $O(1)$ pooled vector;
- **Optional algorithms are bound too**: any component using global attention (e.g. BERT bidirectional encoding, a Transformer decoder on long answers) must be restricted to short sequences or replaced with linear attention / folding structures; otherwise it cannot be the default.

Rationale: inputs can be long documents, retrieved passages, or multi-turn dialogue. Quadratic cost would explode with input length and cancel the gain from vocabulary compression.

## 3. Formalization

### 3.1 Decomposition

$$
p(y \mid x) \;\approx\; p(C \mid x) \cdot p(y \mid C)
$$

- $C = \operatorname{TopK}_k\big(\sigma(f_\theta(x))\big) \cup \mathcal{G}$, i.e. $C$ is a deterministic function $C(x)$ of $x$;
- Module 2 conditions only on $C$ (and an optional $O(1)$ pooled context $h$), modeling $q_\phi(y \mid C) \approx p(y \mid C)$.

Compression necessarily discards word order and part of the context, so this is an **approximation**, not an identity.

### 3.2 Approximation and Its Cost

Compressing $x$ into an unordered word set $C$ is lossy: $C$ carries no word order, negation, or coreference, and the same $C$ can correspond to several valid answers. The factorization is therefore a **deliberate information bottleneck**, not an attempt at an exact probabilistic decomposition.

The cost appears in two places:

- **Insufficient coverage**: answer words outside $C$ cannot be expressed;
- **Generation ambiguity**: the same $C$ with different contexts does not determine a unique answer.

Coverage metrics and triple consistency (Chapter 8) make these costs explicit. Whether to accept the bottleneck depends on how sensitive the task is to word order and context.

### 3.3 Objective

$$
\mathcal{L} = \mathcal{L}_{\text{cov}} + \lambda\, \mathcal{L}_{\text{gen}}
$$

$$
\mathcal{L}_{\text{cov}} = -\frac{1}{N}\sum_{v \in \mathcal{V}} w_v\Big[z_v \log p_v + (1-z_v)\log(1-p_v)\Big],\quad z_v = \mathbb{1}[v \in T(y)]
$$

$$
\mathcal{L}_{\text{gen}} = -\sum_{t=1}^{L} \log q_\phi(y_t \mid x, y_{<t}, C)
$$

where $w_v = \alpha$ (positive) or $1$ (negative). The two modules can be trained separately (staged $\lambda$) or fine-tuned jointly.

## 4. Module 1: Vocabulary Distribution Prediction

### 4.1 Interface

```
input   X: set of word ids (unordered, deduplicated), with a mask
output  s: R^N, p_v = sigmoid(s_v)
select  C = TopK_k(p) ∪ G ∪ (X ∩ V)
```

Design constraints:

- **Permutation-invariant**: shuffling the input order does not change the output;
- **Multi-label**: each word's existence is judged independently, not by a mutually exclusive softmax;
- **Single forward pass**: the whole sentence is processed once, not word by word.

### 4.2 Architecture

```
X (word set)
  │  Embedding (shared vocabulary)
  ▼
per-token linear transform (token MLP, position-wise)
  │  masked mean pooling (permutation-invariant)
  ▼
h ∈ R^d
  │  Linear(d, N)
  ▼
p ∈ [0,1]^N
```

The encoder is deliberately **linear**: each word passes independently through a small MLP, then masked mean pooling; there is no token-to-token attention. Complexity grows linearly with input length, $O(n d^2)$, satisfying 2.4, and pooling keeps the output invariant to word order, matching the "existence only" contract. If stronger interaction is needed, use linear attention (still $O(n)$), but never global $O(n^2)$ attention.

### 4.3 Training

- Target set: $z_v = \mathbb{1}[v \in T(y)]$; repeated words in the answer count once;
- Positives are extremely sparse (about $L/N \approx 0.03\%$), so reweighting or focal loss is required;
- Hard negatives: penalize words highly related to the input but absent from the answer (synonym candidates, same-topic words);
- Checkpoint selection: based on Recall@$k$ and full-sentence coverage, not the BCE value.

### 4.4 Candidate Construction

$$
C = \operatorname{TopK}_k(p) \cup \mathcal{G}
$$

- $\mathcal{G}$: punctuation, EOS, UNK, digit placeholders; optionally high-frequency function words;
- **copy channel (optional)**: only in full-sequence prediction (6.1.2) are input words $X$ merged into $C$; incremental prediction (6.1.1) does not enable it, to keep input words out of the "answer content word" labels;
- threshold augmentation (optional): also include words with $p_v > \tau$, trading a little speed for coverage;
- when batching, pad to $k_{\max}$ within the batch and pass a `cand_mask`.

### 4.5 Key Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VocabularyDistributionPredictor(nn.Module):
    """Module 1: Vocabulary Distribution Prediction.

    Input: unordered word set X (deduplicated word ids)
    Output: logits over the full vocabulary; p_v = sigmoid(logit_v)

    Linear complexity: per-token transform + masked pooling, O(n d^2);
    no attention inside the input.
    """

    def __init__(self, vocab_size: int, d_model: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Position-wise MLP: linear in n
        self.token_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x_ids: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        # x_ids: [B, n]; x_mask: [B, n], 1 = real word
        h = self.token_mlp(self.emb(x_ids))                 # [B, n, d]
        pooled = (h * x_mask.unsqueeze(-1)).sum(1) / x_mask.sum(1, keepdim=True)
        return self.classifier(pooled)                      # [B, N]
```

```python
def coverage_loss(logits, y_ids, pad_id=0, pos_weight=8.0):
    """Module-1 loss: multi-label BCE; target set = answer words (duplicates count once)."""
    target = torch.zeros_like(logits).scatter_(1, y_ids, 1.0)
    target[y_ids == pad_id] = 0.0
    weight = torch.where(target > 0, pos_weight, 1.0)
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)
```

```python
@torch.no_grad()
def select_candidates(logits, k, closure, extra_ids=None, pad_id=0):
    """Build the compressed vocabulary: TopK ∪ closure (optionally plus input words for copy in full-sequence mode).
    Returns variable-length candidate lists."""
    B, N = logits.shape
    k = min(k, N)
    top = logits.topk(k, dim=-1).indices.tolist()          # [B][k]
    out = []
    for b in range(B):
        cand = set(top[b]) | set(closure.tolist())
        if extra_ids is not None:
            cand |= {t for t in extra_ids[b].tolist() if t != pad_id}
        out.append(sorted(cand))
    return out                                             # [B][k_i]
```

> Implementation note: $\lvert C\rvert$ differs across samples; the `collate` function pads to the maximum length in the batch and emits `cand_ids` together with `cand_mask`. `extra_ids` are the padded input word ids (including the deduplicated input set), passed only in full-sequence prediction.

### 4.6 Metrics

- **Recall@k**: $\mathbb{E}\big[\lvert T(y) \cap C\rvert / \lvert T(y)\rvert\big]$, token-level;
- **FullCoverageRate**: $\mathbb{P}[T(y) \subseteq C]$, the fraction of fully covered sentences, which sets the end-to-end ceiling;
- the coverage-vs-$k$ curve and its trade-off with the speedup implied by $k$.

## 5. Module 2: Word-String Generation

### 5.1 Interface

```
input   C: compressed vocabulary + cand_mask
        h: optional pooled context (Module 1 output, O(1))
output  y = (y_1..y_L), every y_t ∈ C, not required to use all of C
```

> Module 2 does not re-read the long input sequence; it conditions only on the compressed vocabulary $C$ and, optionally, an $O(1)$ pooled vector $h$. The unordered input is a design contract: word order and context are deliberately discarded in Module 1, and Module 2 composes from the selection $C$; the resulting ambiguity is measured by 3.2 and Chapter 8.

### 5.2 Architecture

```
C (compressed vocabulary) + h (optional pooled context, O(1))
  │
  ▼
Decoder (self-attention only within answer length L; h added/concatenated)
  │  h_t
  ▼
logits = h_t · E^T   (shared word embeddings)
  │  gather candidate indices of C
  ▼
softmax over k → y_t
```

The output layer no longer projects to $N$; instead it gathers the rows of the shared embedding matrix that correspond to $C$, and the softmax is taken over only the $k$ candidates. When batching, `cand_mask` sets padded positions to $-\infty$.

There is no cross-attention over the long input: the input side is scanned once, linearly, in Module 1. Standard self-attention inside the answer is $O(L^2)$ with $L$ the short answer length; if answers can be long, switch to linear attention or Li-MCTS folding (7.2) to keep 2.4.

### 5.3 Constrained Decoding

$$
q(y_t = v \mid \cdot) = \operatorname{softmax}_{v \in C}\big(h_t^\top e_v\big)
$$

- Greedy: take the highest-scoring word within $C$ at each step;
- Beam: keep $B$ prefixes per step, expand over $C$, with $O(Bk)$ comparisons;
- EOS must be in $C$ (put it in the closure $\mathcal{G}$);
- output length cap $L_{\max}$ and repetition penalty follow standard decoding.

### 5.4 Key Code

```python
class WordStringGenerator(nn.Module):
    """Module 2: Word-String Generation.

    Conditions: ctx, the pooled representation from Module 1 (O(1)), plus the
    compressed vocabulary C.
    Output: a word string constrained to C; softmax dimension = |C|, not |V|.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        layer = nn.TransformerDecoderLayer(
            d_model, nhead=4, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, n_layers)

    def forward(self, y_in, ctx, cand_ids, cand_mask=None):
        # y_in: [B, L]; ctx: [B, 1, d]; cand_ids: [B, k]; cand_mask: [B, k]
        h = self.decoder(self.emb(y_in), ctx)                 # [B, L, d]
        logits = h @ self.emb.weight.T                        # [B, L, N], shared embeddings
        logits = logits.gather(2, cand_ids.unsqueeze(1).expand(-1, h.size(1), -1))
        if cand_mask is not None:                             # pad candidates -> -inf
            logits = logits.masked_fill(~cand_mask.bool().unsqueeze(1), float("-inf"))
        return logits                                         # [B, L, k]
```

```python
@torch.no_grad()
def constrained_decode(model, ctx, cand_ids, cand_mask, bos_id, eos_id, max_len=64):
    """Constrained decoding: search only within the candidate vocabulary C at every step."""
    cand = cand_ids[0].tolist()
    ys = torch.tensor([[bos_id]])
    for _ in range(max_len):
        logits = model(ys, ctx, cand_ids, cand_mask)[0, -1]   # [k]
        nxt = int(cand[int(logits.argmax())])
        ys = torch.cat([ys, torch.tensor([[nxt]])], dim=1)
        if nxt == eos_id:
            break
    return ys
```

> Beam search shares the same interface: replace "take argmax" with "maintain the $B$ best-scoring prefixes".

### 5.5 Training Notes

- **Objective**: teacher-forcing cross-entropy, normalized over $C$ only;
- **Candidate noise**: during training, randomly drop 10–20% of the gold set and add a few distractors, simulating Module 1's misses and extras;
- **Two-stage variant (optional)**: a selection head first predicts a subset $S \subseteq C$ (sigmoid multi-label), then an ordering decoder orders $S$. Direct autoregression already handles "not every word must be used"; the two-stage variant is a fallback for long answers only;
- **Joint fine-tuning**: use a straight-through / Gumbel-softmax estimator for $\operatorname{TopK}$ so Module 2's sequence loss flows back into Module 1.

### 5.6 Metrics

- End-to-end: BLEU / ROUGE / task accuracy, against a full-vocabulary baseline of the same parameter count;
- Constrained perplexity: computed only over $C$, measuring Module 2's own fit;
- Speed: per-sentence latency, tokens/s, peak memory.

## 6. Algorithms for Module 1

Module 1 is essentially "predict a word set from the input". The choice splits into two orthogonal dimensions:

- **What to predict**: incremental / full-sequence / noise-robust;
- **Which model**: MLP / BERT sampling / flow matching.

The two combine freely: e.g. "MLP + incremental" for maximum speed, "BERT encoder + noise-robust" for robustness to noisy inputs.

### 6.1 Three Prediction Regimes

For a training pair $(x, y)$, let $T(\cdot)$ denote the word set. The three regimes differ only in the **label set $Z(x, y)$**:

| Regime | Label set $Z(x, y)$ | Composition of $C$ | Notes |
| --- | --- | --- | --- |
| Incremental | $T_{\text{content}}(y)$: content words of the answer | $Z \cup \mathcal{G}$ | smallest $k$; highest positive density |
| Full-sequence | $T(x) \cup T(y)$ | $Z$ | input words covered by construction; natural copy |
| Noise-robust | $\big(T(x) \cup T(y)\big) \setminus N(x)$ | $Z$ | $C$ stays compact under noisy input |

#### 6.1.1 Incremental Prediction

Predict only the words the answer "may contain", i.e. content words (nouns, verbs, adjectives); function words (of, the, is) and punctuation come from the fixed closure $\mathcal{G}$.

- Training target: $z_v = \mathbb{1}[v \in T_{\text{content}}(y)]$;
- Candidates: $C = \operatorname{TopK}_k(p) \cup \mathcal{G}$;
- Pros: small $\lvert C\rvert$ (typically 64–256), high positive ratio, stable training;
- Difficulty: the content/function boundary depends on tokenization and POS tagging; the closure must cover every function word, otherwise the answer lacks grammatical scaffolding;
- Use cases: short answers, dialogue, well-specified intents.

#### 6.1.2 Full-Sequence Prediction

The label set is the union of the input and output word sets: $Z = T(x) \cup T(y)$.

- Semantically, "all words from input through output are contained in the compressed vocabulary";
- input words enter $C$ automatically (the copy channel), so coverage is automatic whenever the answer reuses input words;
- Difficulty: $\lvert C\rvert$ grows with input length; large documents and retrieved passages carry many stopwords that eat the budget; in the extreme the vocabulary degenerates to "the full vocabulary" and the speedup vanishes;
- Use cases: rewriting, summarization, extractive QA — copy-heavy scenarios.

#### 6.1.3 Noise-Robust Prediction

Treat part of the input words as noise $N(x)$; the label set is the denoised union:

$$
Z = \big(T(x) \cup T(y)\big) \setminus N(x)
$$

The model learns to "drop useless input words and keep useful ones", producing a **denoised full-sequence vocabulary**: it contains both answer words and the input words worth keeping, but no noise words.

Sources of noise labels:

1. **Synthetic noise**: insert irrelevant words or sentences into the input during training and record their positions as noise labels;
2. **Heuristic noise**: stopwords, isolated words with no co-occurrence with the answer, segments unrelated to the answer's meaning;
3. **Learned noise**: a small scorer estimates each input word's contribution to the answer; below-threshold words are treated as noise.

- Pros: in long documents, retrieved passages, and noisy chit-chat, $\lvert C\rvert$ does not grow with the input;
- Difficulty: deleting a key word directly breaks coverage; noise labels can be wrong, requiring joint "coverage + denoising" supervision;
- Relation to full-sequence: full-sequence prediction is the special case $N(x) = \varnothing$.

#### 6.1.4 Comparison

| Aspect | Incremental | Full-sequence | Noise-robust |
| --- | --- | --- | --- |
| Size of $k$ | smallest | grows with input | moderate, controllable |
| Coverage difficulty | medium (content words) | low (copy) | high (may delete) |
| Extra supervision | none | none | noise labels or scorer |
| Main risk | incomplete closure | speedup eaten by input length | deleting key words |
| Best for | dialogue, short answers | rewriting, summarization, extraction | long docs, retrieval-augmented, noisy input |

### 6.2 Three Model Families

#### 6.2.1 MLP: The Simplest Backbone

Average/sum the input word embeddings (or use TF-IDF sparse features directly), pass through a 2–3 layer MLP, and output $N$ logits:

$$
\bar{e} = \frac{1}{n}\sum_{i=1}^{n} E[x_i],\qquad
s = W_2\,\sigma(W_1 \bar{e} + b_1) + b_2 \in \mathbb{R}^{N}
$$

- Cost per sample: $O(n d + d^2 + N d)$, where $N d$ is the output layer and unavoidable;
- Pros: simple, stable, fast; the lower-bound baseline for coverage;
- Cons: no inter-word interaction (the semantics of "not" + "good" is lost); weak compositional generalization;
- Enhancements: low-rank output layer ($W = UV^\top$, $U \in \mathbb{R}^{N \times r}$, $V \in \mathbb{R}^{d \times r}$), feature hashing, hierarchical softmax.

MLP suffices when the set is mostly determined by input frequency and topic (dialogue, templated tasks). Tasks with strong semantic composition need contextual models.

#### 6.2.2 BERT Sampling: Bidirectional Encoding + Iterative Set Sampling

Use a BERT-style bidirectional encoder to obtain contextual representations, then produce the set. Three granularities:

1. **Per-word scoring (marginal approximation)**: score the vocabulary with an MLM head and take top-$k$. Equivalent to "BERT representation + sigmoid head", but each word is judged independently and intra-set dependencies are ignored.
2. **Iterative sampling (MaskGIT-style)**: start from the empty set (or all-masked) and repeat $T$ rounds:
   - predict each candidate word's existence probability;
   - keep the most confident batch of words, leave the rest "undecided";
   - continue the next round conditioned on the decided words.
   Typically $T = 4 \sim 8$. The resulting sets are more coherent and diverse, and $k$ is controllable; sampling is stochastic at inference.
3. **Gibbs sampling**: fix the rest of the set and resample one position at a time; fine-grained but slow.

Training signal: randomly mask the input for denoising training; or add a "set reconstruction" objective on top of multi-label BCE — given some words, predict the rest.

- Pros: strong contextual modeling; can reuse pretrained weights directly;
- Cons: iterative sampling costs $T$ forward passes; results are not guaranteed optimal; thresholds and rounds need calibration on a validation set;
- Complexity: BERT bidirectional encoding is global attention, $O(n^2)$, conflicting with 2.4; it fits short inputs only, or must be replaced by a linear-attention variant, and is a research option rather than the default.

#### 6.2.3 Flow Matching: Generate the Set as a Continuous Distribution

The first two methods output per-word marginals. To **sample a whole set** and preserve the multi-modality of "one input, several plausible word sets", use flow matching to model the set distribution directly.

Let the target set's existence vector be $z \in \{0,1\}^N$ (or $z \in \mathbb{R}^N$ in logit space), noise $z_0 \sim \mathcal{N}(0, I)$, data $z_1 \sim p_{\text{data}}(\cdot \mid x)$. Define the linear interpolation path

$$
z_t = (1 - t)\, z_0 + t\, z_1,\qquad t \in [0, 1]
$$

The velocity field $v_\theta(z_t, t, x)$ regresses $\mathrm{d}z_t/\mathrm{d}t = z_1 - z_0$, with loss

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\, z_0,\, z_1}\Big[\big\| v_\theta(z_t, t, x) - (z_1 - z_0) \big\|^2\Big]
$$

At inference, integrate the ODE from $z_0$ (Euler or RK4, $T$ steps) to obtain $z_1$, then take top-$k$ or threshold to get $C$.

Key engineering issues:

- **Curse of dimensionality**: $N$ can reach $10^5$, so a direct $N$-dimensional flow is too expensive. Feasible options: low-rank flow (integrate in an $r = 64 \sim 256$ latent space, then linearly decode to $N$) or shortlist flow (take $512 \sim 2048$ candidates with a cheap MLP first, then run the flow only on them);
- **Discrete output**: the continuous $z$ needs thresholding or top-$k$ discretization; calibrate the threshold on validation data;
- **Permutation invariance**: the velocity field must not carry positional encodings, or the unordered nature of the set is broken;
- **Coupling**: the pairing of $z_0$ and $z_1$ affects convergence; optimal-transport (OT) coupling can replace independent sampling.

- Pros: explicit generative modeling, good diversity, unaffected by BCE's mode averaging;
- Cons: slow ODE integration, complex implementation, biased discretization; better suited as a quality-ceiling research line than the v1 default.

#### 6.2.4 Recommendations

| Goal | Recommended combination |
| --- | --- |
| Fastest, good enough | MLP + incremental prediction |
| Coverage first, copy-heavy | BERT encoder + full-sequence prediction |
| Long docs / noisy input | BERT encoder + noise-robust prediction |
| Quality-ceiling research | Flow matching (low-rank or shortlist) |

## 7. Algorithms for Module 2

Module 2 generates an ordered word string over the compressed vocabulary $C$. Two routes:

- **Neural autoregression**: Transformer decoder (limited to short answers); mature and stable;
- **Structured search**: Li–Monte Carlo Tree (Li-MCTS); interpretable and potentially more parameter-efficient, but harder to train and infer.

### 7.1 Transformer Decoder

The basic structure is given in 5.2; here are optional variants:

| Variant | Description |
| --- | --- |
| Pooled-conditioning decoder (default) | conditioned on Module 1's $O(1)$ pooled representation $h$; never re-reads the long input |
| Linear-attention decoder | linear attention inside the answer, keeping $O(L)$ |
| Shared embeddings | Output layer reuses input embeddings, gathering candidate rows; fewer parameters |
| Hierarchical vocabulary | Group frequent and content words to further cut softmax cost |
| Trie constraint | For subword vocabularies, expand only over valid prefixes with a trie |

- Pros: parallel training, mature ecosystem, compatible with existing LMs;
- Cons: serial decoding; standard self-attention is $O(L^2)$ over the answer, so long answers need linear attention (otherwise 2.4 is violated).

### 7.2 Li–Monte Carlo Tree (Li-MCTS)

#### 7.2.1 Design Philosophy

Treat "sentence composition" as folding over a **directed graph of binary operation functions**:

- each node $v \in V_G$ is a learnable binary operator $f_v(a, b): \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d$;
- each outgoing edge $e = (v, u)$ is bound to a **value-domain partition region** $R_e \subseteq \operatorname{range}(f_v)$;
- after $f_v$ is computed, the region its output falls into determines which edge routes to the next node $u$;
- edge weights are the partition boundaries and are learnable (scalar quantiles or hyperplanes in embedding space).

In one line: **nodes are operators, edges are value-domain segments, and the path is a data-dependent program.**

#### 7.2.2 Sequence Folding

Read elements $x_1, x_2, \dots, x_n$ one at a time (they can be input words or candidate words):

- **Initial**: $s_2 = f_{\text{init}}(e_{x_1}, e_{x_2})$, and the node becomes $\operatorname{Route}(\text{init}, s_2)$; if the sequence has one element, $s_1 = f_1(e_{x_1})$;
- **Recurrence** ($t \ge 3$):

$$
s_t = f_{node_t}\big(s_{t-1},\, e_{x_t}\big),\qquad
node_{t+1} = \operatorname{Route}(node_t,\, s_t)
$$

- **Readout**: the final state $s_n$ (or each $s_t$) is mapped by a linear readout to a distribution over $C$, or to a scalar score.

$\operatorname{Route}(v, s)$ returns the target node of the outgoing edge whose region contains $s$; partition regions can be scalar quantiles or learnable hyperplanes. Folding costs one binary operator per step ($O(d^2)$), linear in sequence length, with no global attention, so it satisfies 2.4 naturally.

#### 7.2.3 Why Monte Carlo Tree Search

- Routing is discrete (region → node) and non-differentiable;
- the number of possible paths/word orders is enormous ($C$'s subsets × permutations);
- the objective is usually sequence-level (whole-sentence quality), not per-word local optimality.

MCTS provides forward-simulation evaluation, UCT-based exploration, and sequence-level optimization without differentiability. Each tree node keeps visit count $N$ and cumulative reward $W$:

$$
\text{UCT} = \frac{W}{N} + c\sqrt{\frac{\ln N_{\text{parent}}}{N}}
$$

Loop: Select → Expand → Simulate → Backpropagate.

```text
state:  chosen word sequence + current graph node + folding state s
action: pick the next word from C (or an outgoing edge/operator)
sim:    fast rollout (greedy folding) to complete the sequence and score it
final:  readout score of the sequence, or consistency with the target distribution
```

#### 7.2.4 Training Approaches

| Approach | Idea | Difficulty |
| --- | --- | --- |
| Differentiable relaxation | softmax soft routing over regions + temperature annealing; operators trained end-to-end | medium |
| Straight-through | hard routing forward, soft-routing gradients backward | medium |
| Reinforcement learning | hard routing; REINFORCE / policy-gradient updates of routing decisions | high |
| Two-stage | teacher-forced folding on gold sequences first, then MCTS fine-tuning | high |

The graph can be hand-designed (a few operators: add, multiply, gate, small MLP) or discovered by neural architecture search.

#### 7.2.5 Adaptive Graph Growth

Nodes and edges are not fixed before training; they grow dynamically during training and operation based on usage:

- **Start small**: the initial graph can be tiny (e.g. only the initial function and a few nodes); nodes and edges are added on demand;
- **Hot-node out-degree expansion**: track how often each node's function is invoked and how often each edge is hit; the more a node is used, the larger its out-degree — partition its value range more finely and add branches so frequent paths are modeled in more detail;
- **Random new connections**: when growth is needed, connect a new edge to an existing node with some probability (reusing an already-learned subgraph), or create a new node with some probability (opening a new branch);
- **Capacity limits**: a per-node out-degree cap and a total edge budget; periodically prune low-usage edges so the graph does not blow up;
- **Split policy**: when an out-degree expands, prefer splitting value ranges with dense samples so new branches receive enough training signal.

This lets graph capacity follow the data distribution: hot regions become finer, cold regions stay sparse. It is also the structural side of online learning (9.3).

#### 7.2.6 Pros and Cons

- Pros:
  - shared computation graph; possibly more parameter-efficient than a Transformer of the same capacity;
  - value-domain routing is piecewise; paths are interpretable (tree/automaton-like);
  - inference can prune by value range and jump to relevant subgraphs.
- Cons:
  - joint optimization of graph and routing is hard;
  - MCTS needs many simulations; latency is higher than autoregression;
  - folding is inherently serial, hard to parallelize on GPUs;
  - errors accumulate over long sequences; routing jitter can destabilize paths.

#### 7.2.7 Comparison with Transformer

| Aspect | Transformer | Li-MCTS |
| --- | --- | --- |
| Training parallelism | high | low (serial folding) |
| Routing/structure | continuous attention | discrete value-domain routing |
| Interpretability | medium | higher |
| Inference latency | $O(kd)$ per step | $O(d^2)$ per step + MCTS simulations |
| Engineering maturity | high | low (research) |
| Best for | general generation | strongly structured, interpretable, small-model settings |

## 8. Optimization Objective: Triple Consistency

### 8.1 The Three Elements

For a sample $(x, y)$:

- $A = T(y)$: the future-content word set given by the training data;
- $B = T(\hat{y})$: the word set actually produced by word-string generation;
- $C = C(x)$: the compressed vocabulary.

Three constraints form a closed loop:

1. **Coverage**: $A \subseteq C$ — the future content must be contained in the compressed vocabulary;
2. **Generation consistency**: $B \approx A$ — the word-string prediction must agree with the future content;
3. **Containment**: $B \subseteq C$ — the word-string prediction must be contained in the compressed vocabulary.

Constraint 3 is guaranteed by construction under constrained decoding ($B$ can only be drawn from $C$); what really needs optimizing is constraints 1 and 2 and their mutual tension.

### 8.2 Why a Triple

Each constraint alone can be gamed:

| Optimize only | Cheat | Consequence |
| --- | --- | --- |
| Coverage $A \subseteq C$ | Set $C = \mathcal{V}$ | coverage 100%, speedup zero |
| Generation consistency $B \approx A$ | Ignore $C$, fall back to a plain LM | the framework loses its meaning |
| Containment $B \subseteq C$ | Make $C$ huge, or output an empty sequence | constraint trivially satisfied, quality collapses |

Only the triple, plus a budget on $\lvert C\rvert$, prevents trivial solutions. This is the same idea as cycle consistency: let three quantities corroborate each other instead of optimizing them independently.

### 8.3 Formalization

- Coverage constraint: $\mathbb{E}[v_1] \le \varepsilon_1$, with $v_1 = \mathbb{1}[A \not\subseteq C]$;
- Generation consistency: $\mathcal{L}_{\text{gen}} = -\log q(y \mid x, C)$ (teacher forcing), or sequence similarity $s(B, A) \ge \tau$;
- Containment constraint: $v_3 = \mathbb{1}[B \not\subseteq C] \equiv 0$ (hard mask);
- Capacity constraint: $\lvert C\rvert \le k$.

Total objective:

$$
\mathcal{L} = \mathcal{L}_{\text{gen}} + \lambda\,\mathcal{L}_{\text{cov}} + \mu\,\frac{\lvert C\rvert}{k} + \nu\, V_{\text{gen}},
\qquad V_{\text{gen}} = 1 - s(B, A)
$$

Constrained formulation:

$$
\min\ \mathcal{L}_{\text{gen}} + \mu\, \mathbb{E}\frac{\lvert C\rvert}{k}
\quad \text{s.t.}\quad
\Pr[A \subseteq C] \ge 1 - \varepsilon_1,\ \
\Pr[s(B, A) \ge \tau] \ge 1 - \varepsilon_2,\ \
B \subseteq C
$$

In practice a penalty form is more common: penalize coverage shortfalls and budget overruns; tune $\lambda, \mu, \nu$ on validation data or treat them as adaptive Lagrange multipliers.

### 8.4 Training Strategy

- **Alternating optimization (hard-EM style)**:
  - E-step: freeze the generator, update Module 1 to minimize $\lvert C\rvert$ subject to coverage $\ge 1 - \varepsilon_1$;
  - M-step: freeze Module 1, update Module 2 to maximize $\log q(y \mid x, C)$.
- **End-to-end**: straight-through top-$k$ / Gumbel-softmax to backpropagate the sequence loss into Module 1;
- **Curriculum**: first train Module 2 to near-ceiling on gold $C$ (so constraint 2 is not held back by constraint 1), then switch to predicted $C$;
- **Lagrange multipliers**: turn the coverage constraint into an adaptive $\lambda$ and avoid manual tuning.

### 8.5 Metrics

- **Triple-consistency rate**: $R = \Pr\big[v_1 = 0 \wedge s(B, A) \ge \tau \wedge v_3 = 0\big]$;
- coverage (Recall@$k$ / FullCoverageRate);
- generation quality (BLEU / ROUGE / EM);
- mean $\lvert C\rvert$ and inference speed.

The triple-consistency rate is the single most faithful metric for this framework: it simultaneously requires "the vocabulary is sufficient", "the generated sentence is correct", and "every generated word is in the vocabulary".

## 9. Training Pipeline

### 9.1 Staged Pipeline

```
Stage 0  Data construction
        x → deduplicated word set; y → word sequence; profile the distribution of |T(y)|
Stage 1  Module 1
        multi-label BCE with positive weighting; select by Recall@k / coverage
Stage 2  Module 2 (gold C)
        cross-entropy + candidate noise; this stage establishes the quality ceiling
Stage 3  Module 2 (predicted C)
        freeze Module 1; continue training on its C (self-distillation)
Stage 4  Optional joint fine-tuning
        straight-through top-k + sequence loss; small learning rate
```

Why this order: if Module 2 faces Module 1's predicted set from the start, it learns "compensation strategies" for random misses and masks Module 1's coverage problems; training on gold first, then predicted, lets the two error sources be located separately.

### 9.2 Input Noise Injection

Training inputs are not always clean:

- Insert random noise words at **random positions** in the input sequence (sampled uniformly from the vocabulary or from a dedicated noise pool);
- The insertion probability increases with position: **the closer to the sentence end, the denser the noise**;
- The noise ratio follows a curriculum, ramping from 0 to a target level (e.g. 10–20%).

Rationale:

1. Both modules see "extra words", learn to ignore noise and preserve coverage, pairing naturally with noise-robust prediction (6.1.3);
2. Generation starts right after the input end, so end-of-sentence noise interferes most with the first output word and needs a denser training signal;
3. Noise injection is data augmentation and mitigates overfitting on small data.

Implementation notes:

- Noise words must not coincide with answer words, or coverage labels get polluted;
- Record insertion positions per sample for noise-robust prediction;
- Inject noise during training only; keep inference inputs clean.

### 9.3 Online (Real-Time) Learning

Training philosophy: the model should not be frozen after one unified large-scale initial training; it should keep updating **in real time** as new data arrives.

- **Online updates**: apply small-step updates as new samples arrive (small learning rate; EWC-style regularization when needed);
- **Replay buffer**: retain a small set of old samples to mitigate catastrophic forgetting;
- **Adaptive structure**: graph structure and candidate statistics grow with use (see 7.2.5);
- **Monitoring and rollback**: track online metrics and roll back to the previous checkpoint on degradation;
- **Division of labor with offline training**: offline large-scale training provides the initial state and a stable base distribution; online learning handles continuous adaptation.

Risks: catastrophic forgetting, polluted online data, lagging evaluation. Mitigations: replay + small learning rate + periodic full refresh + data-quality filtering.

## 10. Inference and Complexity

```
input x
  │ Module 1 (single forward pass)
  ▼
s ∈ R^N → C (TopK + closure, optional copy)
  │
  ▼
Module 2 (autoregressive, vocabulary = C)
  ▼
answer y
```

- linear in input length: Module 1 is a single $O(n d^2)$ scan; Module 2 is independent of input length;
- Module 1 costs one $O(N d)$ output layer per sentence;
- Module 2 costs $O(k d)$ per step;
- the speedup grows with $L$;
- batching candidates and masks for Module 2 is the main engineering complexity.

## 11. Evaluation

### 11.1 Data

Cover three task shapes:

- dialogue (e.g. DailyDialog): the copy channel pays off;
- summarization (e.g. CNN/DailyMail, LCSTS): content words are determined by the input;
- QA (e.g. SQuAD): answer words often appear in the input.

### 11.2 Baselines

1. a standard Transformer LM with the same parameter count and data;
2. a standard LM with per-step top-$k$ sampling (distinguishing "per-step truncation" from "whole-sentence compressed vocabulary");
3. adaptive softmax (another route to cheaper output layers).

### 11.3 Metrics

| Category | Metrics |
| --- | --- |
| Coverage | Recall@k, FullCoverageRate, $k$-coverage curve |
| Quality | BLEU / ROUGE / EM / task accuracy |
| Language modeling | constrained perplexity |
| Efficiency | latency, throughput, output-layer FLOPs, memory |
| Ablation | $k$, closure, copy, candidate noise, joint training |

### 11.4 Key Experiments

- coverage-vs-speedup curve: find the knee in $k$;
- quality comparison with baselines at equal latency;
- Module 1's share of cost across input lengths.

## 12. Limitations and Risks

1. **Coverage is not 100%**: numbers, proper names, and rare words easily fall outside $C$; v1 mitigates with thresholds and the copy channel and reports coverage as the main risk metric.
2. **Semantic loss from unordered input**: Module 1 models sets, so negation, coreference, and subject–object relations are dropped at compression time; Module 2 never re-reads the input and composes from $C$ and the pooled $h$. This is a design trade-off whose cost is measured by coverage and triple consistency.
3. **Fixed $k$ budget**: long answers and enumerations need larger $k$; a fixed $k$ cannot serve both; an adaptive $k$ predicted from length is a direction.
4. **Function words eat the budget**: the closure and function words are almost always present and squeeze content words.
5. **Two-stage error accumulation**: when Module 1 misses a word, Module 2 cannot recover it; joint training only mitigates.
6. **Module 2 capacity**: a small-vocabulary decoder has limited long-range structure modeling (nested clauses).
7. **Tokenization granularity**: words or BPE; Chinese has no spaces, so word boundaries need a convention.

## 13. Related Work

- **Copy / pointer-generator networks**: allow copying input words during generation, but the output layer still covers the full vocabulary; this framework merges input words into the compressed vocabulary only in full-sequence prediction mode.
- **Adaptive softmax**: hierarchical structures cut output-layer cost, but decoding stays word-by-word and the search space is unchanged; this framework shrinks both the output layer and the search space.
- **kNN-LM / RETRO**: retrieve from external memory or corpora, solving "where does knowledge come from"; this framework solves "how large is the candidate set".
- **Constrained decoding**: Module 2 is exactly "generation under a vocabulary constraint", with the constraint supplied dynamically by Module 1.
- **Set2Seq**: set-to-sequence modeling; shares the "unordered input → ordered output" form with this framework.
- **Speculative decoding**: draft with a small model and verify with a large one; orthogonal and combinable.
- **Non-autoregressive / diffusion LMs**: parallel generation; the hard parts are order and consistency; this framework keeps autoregression but compresses the vocabulary.
- **Masked generation / flow matching**: MaskGIT-style iterative sampling and flow matching model generation as sets or continuous distributions; Chapter 6 lists both as options for Module 1.
- **Cycle consistency**: lets multiple constraints corroborate each other; the triple consistency in Chapter 8 borrows this idea.

## 14. Roadmap

| Version | Goal | Criterion |
| --- | --- | --- |
| v0.1 | Module 1 coverage validation | FullCoverageRate ≥ 95% on small-scale data |
| v0.2 | End-to-end quality vs baselines | at equal latency, quality ≥ 95% of baseline |
| v0.3 | Escape hatch | no silent quality loss when coverage fails |
| v0.4 | Joint training | stable gains over separate training |
| v1.0 | Full experiments | three task shapes, complete ablations, reproducible |

## 15. Changelog

| Date | Change |
| --- | --- |
| 2026-11-26 | Initial version: module decomposition and exactness condition; interfaces, code, evaluation |
| 2026-11-26 | v1.1: added Chapters 6 (Module 1 algorithms: three prediction regimes × MLP/BERT sampling/flow matching), 7 (Module 2 algorithms: Transformer/Li-MCTS), 8 (triple-consistency objective) |
| 2026-11-26 | v1.2: consolidated three same-name copies into one document under the single-copy rule; drafts remain in git history only |
| 2026-11-26 | v1.3: input noise injection and online learning (9.2, 9.3); adaptive Li-MCTS graph growth (7.2.5) |
| 2026-11-26 | v1.4: added the linear-complexity design principle (2.4); Module 2 no longer reads the ordered input and depends only on the compressed vocabulary plus an $O(1)$ pooled context; replaced the exactness claim with an information-bottleneck framing; copy channel limited to full-sequence mode |
