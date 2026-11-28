# Implicit Information Representation in Transformers (Extended)

> Stage: **In Progress** (~8 pages) · Status: **WIP** · Last updated: 2026-11-28
> Language: [中文](../../扩展中途版/Transformer的隐式信息表示.md) · English

## 1. Observation: Same Structure, Diverging Semantics

Consider three sentences:

- I am a cat.
- I am a doctor.
- I am a driver.

Their syntactic structure is identical (the template "I am + noun"), yet semantically the latter two are much closer: doctor and driver are **professions**, while cat is a **species**.

If the KV projection of a Transformer really encodes semantic structure, one may conjecture:

- "I am a doctor" and "I am a driver" have **more similar shapes** under the KV projection;
- "I am a cat" is only partially similar to them (similar structure, distant semantics);
- unrelated sentences do not share this shape at all.

Here "shape" is not an absolute coordinate but the **relative configuration** of a sentence's representation in KV space: angles, distances, and subspace structure between token vectors. It should be invariant under rotation and translation.

## 2. KV Projections and "Shape"

### 2.1 KV Projections

Each Transformer layer projects hidden states $X$ into keys and values:

$$
K = X W_K,\qquad V = X W_V
$$

A sentence thus yields a set of key vectors $\{k_t\}$ and a set of value vectors $\{v_t\}$. Attention is determined by their inner products, and the output is a weighted sum of values.

### 2.2 Measuring Shape

View the sentence representation as a point cloud. Its shape can be described by the centered Gram matrix:

$$
G_{ij} = \big\langle k_i - \bar{k},\; k_j - \bar{k} \big\rangle
$$

Distances between two shapes can use:

- **Procrustes distance**: orthogonal alignment followed by a Frobenius comparison;
- **principal angles** between subspaces spanned by the two point clouds;
- **CKA**: linear or kernel similarity between representations.

All of these are insensitive to a global rotation and translation, matching the intuition of "shape".

### 2.3 Conjectures

- **G1**: closer semantics ⇒ smaller KV shape distance;
- **G2**: same structure but different semantics ⇒ only partial shape similarity;
- **G3**: unrelated sentences share no stable shape.

These are directly testable with template sentences (Section 6).

## 3. Uniqueness of the Prior-Free Embedding: the Regular Simplex

### 3.1 The Prior-Free Assumption

Assume words are embedded into Euclidean space with **no prior at all**: all words are equivalent under symmetry, and no word is "naturally closer" to another.

### 3.2 Theorem and Proof

**Theorem 1**: If $n$ words are pairwise equidistant, they must be the vertices of a regular $(n-1)$-simplex; the embedding is unique up to isometry.

**Proof**: Let the embedding be $v_1, \dots, v_n \in \mathbb{R}^d$ with all pairwise distances equal to a constant $d_0$, and centroid $\bar{v} = \frac{1}{n}\sum_i v_i$.

For any $i$, the identity

$$
\sum_{j=1}^{n} \|v_i - v_j\|^2 = n\|v_i - \bar{v}\|^2 + \sum_j \|v_j - \bar{v}\|^2
$$

holds. Since all distances are equal, the left side is $(n-1)d_0^2$, and both terms on the right are equal for all $i$; hence $\|v_i - \bar{v}\|^2$ is constant:

$$
\|v_i - \bar{v}\|^2 = \frac{(n-1)d_0^2}{2n}
$$

Therefore, for $i \ne j$,

$$
\langle v_i - \bar{v},\, v_j - \bar{v}\rangle
= \frac{s_i + s_j - d_0^2}{2}
= -\frac{d_0^2}{2n}
$$

The centered Gram matrix $G$ is thus

$$
G = \frac{d_0^2}{2} I - \frac{d_0^2}{2n} J
$$

where $J$ is the all-ones matrix. Its eigenvalues are $d_0^2/2$ (multiplicity $n-1$) and $0$ (multiplicity 1), with rank $n-1$. Hence the points lie in an $(n-1)$-dimensional affine subspace, and the Gram matrix is exactly that of a regular simplex; a Gram matrix determines the point set up to isometry. $\square$

### 3.3 Corollaries

- If the embedding is not equidistant, some pair of words is closer than another pair — this is equivalent to introducing a **prior**;
- so under "no prior", the embedding of $n$ words has essentially one choice (up to isometry): the regular simplex;
- the simplex embedding has dimension $n-1$. Reducing the dimension necessarily requires a prior (which words should be closer, which dimensions can be dropped).

## 4. The Projection Family under the Simplex Embedding

### 4.1 Setup

Take the standard simplex vertices $e_1, \dots, e_n$ (the standard basis of $\mathbb{R}^n$). A vertex subset $S \subseteq \{1, \dots, n\}$ spans the sub-simplex

$$
\operatorname{aff}(S) = \Big\{ \sum_{i \in S} \lambda_i e_i \;:\; \sum_i \lambda_i = 1 \Big\}
$$

of dimension $|S| - 1$.

### 4.2 Proposition

**Proposition 2**: Let the integer $k \ge 1$. For any two vertex subsets $S, T$ with $|S|, |T| > k+1$, there is a family of low-dimensional projections $\mathcal{F}(S, T)$ (projections into $\mathbb{R}^k$) such that:

1. for every $P \in \mathcal{F}(S, T)$, $P(\operatorname{aff}(S)) = P(\operatorname{aff}(T))$;
2. for any vertex subset $U$ with $|U| > k+1$, if $S \nsubseteq U$ and $T \nsubseteq U$, then some $P \in \mathcal{F}(S, T)$ satisfies $P(\operatorname{aff}(U)) \ne P(\operatorname{aff}(S))$.

> Note: $|S| > k+1$ ensures that projecting into $\mathbb{R}^k$ does not exceed the sub-simplex dimension and leaves enough vertices for the construction. Excluding $S \subseteq U$ or $T \subseteq U$ is necessary: in these cases $P(\operatorname{aff}(U)) \supseteq P(\operatorname{aff}(S))$, and all images lie in $\mathbb{R}^k$, so the images must coincide (see 4.4).

### 4.3 Construction and Proof

Take the regular simplex $Q = \{q_0, \dots, q_k\}$ in $\mathbb{R}^k$ ($k+1$ vertices). For any pair of surjections

$$
\pi_S: S \to Q,\qquad \pi_T: T \to Q
$$

define a linear map $P$ by, on the standard basis,

$$
P(e_s) = q_{\pi_S(s)}\ (s \in S),\qquad
P(e_t) = q_{\pi_T(t)}\ (t \in T),
$$

with arbitrary values on vertices outside $S \cup T$ (e.g. all mapped to some $q$). Extend linearly to $\mathbb{R}^n$. All such $P$ form the family $\mathcal{F}(S, T)$.

**(1) Coinciding images**: $P(\operatorname{aff}(S))$ is the convex hull of $\{q_0, \dots, q_k\}$, i.e. $\operatorname{aff}(Q)$; likewise $P(\operatorname{aff}(T)) = \operatorname{aff}(Q)$. So the images are equal.

**(2) Distinguishing $U$**: since $S \nsubseteq U$, choose $s^* \in S \setminus U$; since $T \nsubseteq U$, choose $t^* \in T \setminus U$. Fix $q^* \in Q$ and set

$$
\pi_S(s^*) = \pi_T(t^*) = q^*,
$$

and map the remaining elements of $S$, the remaining elements of $T$, and vertices outside $S \cup T$ into $Q \setminus \{q^*\}$. This is a valid surjection: $q^*$ already has preimages $s^*, t^*$, and the other $k$ vertices are covered by $S \setminus \{s^*\}$ (at least $k$ elements).

Then no vertex of $U$ coming from $S$ or $T$ maps to $q^*$ (they are not $s^*, t^*$), and the remaining vertices of $U$ are also arranged to avoid $q^*$. Hence $P(\operatorname{aff}(U))$ does not contain $q^*$, while $P(\operatorname{aff}(S)) = \operatorname{aff}(Q)$ does; they differ. $\square$

### 4.4 Degenerate Case: Containment Is Invisible under Such Projections

If $S \subseteq U$, then $\operatorname{aff}(U) \supseteq \operatorname{aff}(S)$, so for any linear $P$, $P(\operatorname{aff}(U)) \supseteq P(\operatorname{aff}(S))$; and all images lie in $\mathbb{R}^k = \operatorname{aff}(Q)$, so $P(\operatorname{aff}(U)) = \operatorname{aff}(Q) = P(\operatorname{aff}(S))$. In other words, **containment cannot be distinguished by this class of projections**.

This is not a defect of the proof but a direct consequence of the simplex symmetry: the simplex embedding itself carries no semantic distinction beyond "which subsets contain which".

### 4.5 Implication

- in the regular simplex, any two sub-simplices can be flattened into the same shape by a family of projections, while the same family contains projections separating a third one (as long as it does not contain the first two);
- the symmetry is so strong that projections can "erase" any two subsets — **shape similarity alone does not constitute semantic distinction**;
- therefore the "profession shape vs species shape" observed in a Transformer can only be the result of training sculpting the symmetric simplex geometry, not something the embedding geometry provides for free.

## 5. Implications for Implicit Representations in Transformers

### 5.1 Conjecture G4: KV Shape Is an Implicit Semantic Representation

Training gradually sculpts the prior-free simplex geometry into a layered shape: profession sentences form a similar low-dimensional configuration in KV space, species sentences form another cluster. This matches the observation in Section 1.

### 5.2 Possible Training Methods

1. **Shape contrastive learning**: pull Procrustes distances between same-semantic-class sentence pairs closer, push different classes apart;
2. **Hierarchical semantic supervision**: use labels such as "species / profession / other" as a prior to guide embeddings or KV projections;
3. **Simplex initialization**: initialize word embeddings with the regular simplex (equidistant, prior-free), then let training break the symmetry;
4. **Dimensionality reduction**: reduce KV dimension while preserving shape separability (e.g. semantic triplet accuracy), choosing the dimension by performance;
5. **Shape regularization**: constrain the KV Gram matrix to encourage intra-class stability and inter-class separation.

### 5.3 Cautions

- KV shape depends on layer, attention head, and positional encoding; sentence-level shape needs a defined aggregation;
- shape similarity is not causal contribution and needs intervention experiments;
- semantic labels are themselves human priors, at odds with the "no prior" ideal.

## 6. Testable Predictions and Experiments

- **E1 template sentences**: generate a set of "I am X" sentences, measure the KV shape distance matrix, and check whether professions vs species cluster;
- **E2 layer comparison**: shallower layers may reflect structure, deeper layers semantics;
- **E3 random vs trained**: is the shape of random initialization closer to a simplex (equidistant)?
- **E4 simplex initialization**: compare with standard initialization and see whether semantic clustering improves or degrades;
- **E5 dimensionality curve**: project KV to $d'$ dimensions and measure shape-classification accuracy versus $d'$;
- **E6 shape training**: after adding a shape contrastive loss, does the distance matrix follow the semantic hierarchy?

## 7. Limitations

- **The simplex assumption is strong**: real word embeddings are shaped by data, training, and optimization, far from equidistant;
- **Shape is not uniquely defined**: Gram, subspace, and Procrustes measures each have biases;
- **Aggregation**: sentence-level shape depends on how tokens are pooled (mean, attention pooling, etc.);
- **Confounders**: positional encodings, layer normalization, and residuals all change the geometry;
- **Causality**: shape similarity is only correlation, and does not directly show that the model "reasons by shape".

## 8. Related Work

- Word-embedding geometry: anisotropy and cone effects;
- Representation similarity: CKA, Procrustes alignment, subspace methods;
- Equidistant codes and tight frames: regular simplices, equiangular tight frames;
- Superposition and the linear representation hypothesis;
- Shape and symmetry in geometric deep learning.

## 9. Roadmap

| Version | Goal | Criterion |
| --- | --- | --- |
| v0.1 | Shape-measurement tooling | Procrustes / subspace distances available |
| v0.2 | Template-sentence experiment | professions vs species are shape-separable |
| v0.3 | Complete Proposition 2 | covers general $k$ and all degenerate cases |
| v0.4 | Shape contrastive training | distance matrix follows the semantic hierarchy |
| v1.0 | Full report | positive and negative results, ablations, reproducible |

## 10. TODO

- [ ] Define a standard aggregation for sentence-level KV shape
- [ ] Measure "I am X" shape distances on open models
- [ ] Check whether random initialization is nearly equidistant
- [ ] Implement a shape contrastive loss
- [ ] Generalize Proposition 2 to more general complexes

## 11. Glossary

| Term | Meaning |
| --- | --- |
| KV projection | the linear maps producing keys/values in attention |
| Shape | the geometric configuration of a representation point cloud, invariant to rotation/translation |
| Regular simplex | a simplex whose vertices are pairwise equidistant |
| Procrustes distance | minimum Frobenius distance after orthogonal alignment |
| Principal angles | the sequence of angles between two subspaces |

## 12. Changelog

| Date | Change |
| --- | --- |
| 2026-11-28 | First draft: shape observation, simplex uniqueness, the projection-family proposition, and implications for Transformers |
