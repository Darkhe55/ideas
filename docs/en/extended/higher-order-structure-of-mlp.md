# Higher-Order Structure of MLP (Extended)

> Stage: **In Progress** (~8 pages) · Status: **WIP** · Last updated: 2026-11-28
> Language: [中文](../../扩展中途版/MLP的结构升阶.md) · English

## 1. The Idea

Standard MLP weights are usually learned and stored as a huge matrix. Change the perspective:

- **Linear-map weights** can be seen as **edge weights** of a graph;
- **Biases and activation functions** can be seen as **node weights** (constant terms and nonlinear selection).

Compared with a function expansion or decomposition, linear weights correspond to **first-order terms**, and biases plus activations correspond to **constant terms**. If the analogy holds, we can keep adding orders:

- second-order parameters = **face weights** (2-dimensional simplices);
- third-order parameters = **volume weights** (3-dimensional simplices);
- and so on.

That is, upgrade the graph structure of the MLP to a **simplicial complex**, where weights live on simplices of every dimension. The language of **homology / cohomology** can then describe redundancy and invariant structure in the weights, and low-order information may compress high-order weights.

**Conjecture**: there is a suitable representation space (possibly a Hilbert space) in which higher-order weights shrink and converge to zero; if so, only lower-order weights are needed for a high-precision representation.

But this is a conjecture, not a fact: high- and low-order information may be **complementary** — high-order terms carry what low-order ones cannot generate (the cohomology classes in 4.2 are exactly such irreducible information), and discarding them directly loses accuracy. Computation is also ordered (4.3), and the representation space need not be complete or bounded (5.4). The compression conclusion is therefore a **conditional hypothesis** that must be tested.

## 2. The Weight Structure of a Standard MLP

### 2.1 Computational-Graph View

$$
h^{(l+1)} = \sigma\big(W^{(l)} h^{(l)} + b^{(l)}\big)
$$

Treat each computational unit as a node:

- **Nodes**: bias $b_i$ and activation $\sigma$ set the baseline response and the nonlinear shape;
- **Edges**: $W_{ij}$ sets the linear coupling strength between units;
- the whole network is a weighted directed graph, bipartite between adjacent layers.

A standard MLP has only 0- and 1-dimensional structure: nodes and edges.

### 2.2 Function-Expansion View

Treat the network output as a function $f(x)$ of the input $x$ and expand it near the origin:

$$
f(x) = c + \sum_i a_i x_i + \sum_{i<j} b_{ij} x_i x_j + \sum_{i<j<k} c_{ijk} x_i x_j x_k + \cdots
$$

- the constant term $c$ ↔ the baseline from bias/activation;
- first-order terms $a_i$ ↔ linear-map weights;
- second-order terms $b_{ij}$ ↔ pairwise interaction between input coordinates, i.e. an **edge** (1-simplex);
- third-order terms ↔ three-way interaction, i.e. a **face** (2-simplex);
- and so on.

> Note: the polynomial expansion holds only in an analytic or local sense. Piecewise-linear networks such as ReLU have no global polynomial expansion and need a piecewise treatment; the expansion here mainly serves to define "order" and measure per-order contributions, not to claim that the network equals a low-order polynomial.

> Terminology: in this document "order" means the dimension of the simplex. Order 0 = node (constant term), order 1 = edge (linear term), order 2 = face (three-vertex interaction), order 3 = volume (four-vertex interaction). If one counts by polynomial degree, a $k$-simplex corresponds to a degree-$(k+1)$ term; the two differ by one, and "order = simplex dimension" is used throughout.

### 2.3 A Caveat

In an MLP, $W$ connects units of adjacent layers, not input coordinates. The two sections above therefore describe two different "graphs":

- **Parameter graph** (computational graph): nodes are units, edges are weights;
- **Expansion graph** (interaction graph): nodes are input coordinates, edges are pairwise interactions.

The starting point of the upgrade is to unify both on one complex: each simplex is simultaneously a combination of computational units and an interaction term. Making this correspondence rigorous is one of the core difficulties (see Section 8).

## 3. From Graph to Complex

### 3.1 Simplicial Complex

Given a vertex set $V$, a $k$-dimensional simplex $\sigma = \{v_0, \dots, v_k\}$ is a combination of $k+1$ vertices; a simplicial complex $K$ is a set of simplices closed under taking subsets.

- 0-simplex: vertex (node);
- 1-simplex: edge (two vertices);
- 2-simplex: triangle (three vertices);
- 3-simplex: tetrahedron (four vertices).

A standard MLP corresponds to a $K$ containing only 0- and 1-simplices; the upgrade adds higher-dimensional simplices to $K$ and places weights on each simplex.

### 3.2 Weights and Functions on the Complex

Assign a weight $w_\sigma \in \mathbb{R}$ (or a vector/tensor) to each simplex $\sigma$, and define a basis function $\varphi_\sigma(x)$ (e.g. $\prod_{v \in \sigma} x_v$, or a nonlinear function assembled from vertex representations). The network output is

$$
f_w(x) = \sum_{\sigma \in K} w_\sigma\, \varphi_\sigma(x)
$$

- 0-dimensional terms: bias/baseline;
- 1-dimensional terms: linear maps;
- 2-dimensional terms: three-body interaction (face weights);
- 3-dimensional terms: four-body interaction (volume weights);
- and so on.

Weights are no longer just a matrix; they are quantities on chains distributed over simplices of every order.

### 3.3 Implicit Higher-Order Information in Depth vs. Low-Order Loss

The composition of many layers can itself encode higher-order information: even if each layer only has a linear map and a pointwise activation, composing layers produces higher-order interactions among input coordinates. In other words, high-order terms do not necessarily need explicit parameters — **depth can represent them implicitly**.

But this raises a problem: **low-order information is easily lost through layer-by-layer transformation**.

- each layer transforms the representation, so original coordinates and linear terms are repeatedly re-encoded and may be attenuated, aliased, or overwritten;
- without cross-layer paths, the network must reconstruct low-order information at every layer, wasting capacity and failing more easily in deep stacks;
- typical symptoms are rank collapse, signal attenuation, and insensitivity to input detail.

**Residual and U-Net structures are explicit ways to preserve low-order information**:

- residual connections $h_{l+1} = h_l + F(h_l)$ provide an identity path so low-order components pass through unchanged (He et al., 2016);
- U-Net-style skip connections send shallow (low-order, fine-grained) representations directly to deep layers, preserving multi-scale information (Ronneberger et al., 2015);
- in the complex view, these skips add new 1-simplices (or even higher cells) between non-adjacent layers; the complex is no longer a simple layered bipartite graph, and its homology changes accordingly.

Implication for the compression conjecture: if higher-order information is already implicit in depth, fewer explicit high-order parameters may be needed and the compression room is larger; but only if low-order information survives the depth — which is exactly what residual/U-Net structures address. Otherwise truncating low-order terms or compressing parameters loses accuracy directly.

## 4. Homological Representation of Weights

### 4.1 Chain Complex and Cochains

A simplicial complex gives a chain complex (Hatcher, 2002):

$$
\cdots \to C_{k+1} \xrightarrow{\partial_{k+1}} C_k \xrightarrow{\partial_k} C_{k-1} \to \cdots,\qquad \partial_k \partial_{k+1} = 0
$$

where $C_k$ is spanned by $k$-simplices. Weights are $k$-cochains $w^k \in C^k = \operatorname{Hom}(C_k, \mathbb{R})$, with coboundary $\delta_k = \partial_{k+1}^\top$.

### 4.2 Exact, Closed, and Cohomology Classes

- **Exact cochains**: $w = \delta u$. The weight is fully determined by a lower-order potential $u$, so it can be compressed away — store only $u$;
- **Closed cochains**: $\delta w = 0$. The weights are "consistent" on the complex, with no local contradiction;
- **Cohomology classes**: $H^k = \ker \delta_k / \operatorname{im} \delta_{k-1}$. They capture the part of the weights that **cannot** be generated from lower-order potentials — the genuinely high-order information.

The standard Hodge decomposition gives the orthogonal direct sum (Lim, 2020):

$$
C^k = \operatorname{im}\delta_{k-1} \oplus H^k \oplus \operatorname{im}\delta_k^{*}
$$

where $H^k = \ker\delta_k \cap \ker\delta_{k-1}^{*}$ is the space of harmonic representatives and $\delta_k^{*}$ is the adjoint of $\delta_k$. Any weight then splits into three parts: the gradient of a lower-order potential $\delta u$, a harmonic part $h$, and a coexact part $\delta_k^{*} v$ generated by a higher-order potential. The exact part is generated by lower-order potentials, the harmonic part is irreducible high-order information, and the coexact part by higher-order potentials.

### 4.3 Ordered Computation: Abelian Homology Is Not Enough

Chain complexes are built on free abelian groups (or real vector spaces) with $\partial^2=0$; this is a **commutative** structure: it records which simplices appear with which coefficients, but not the **order** in which the operations happen.

Computation in a neural network is ordered: $W^{(l)}$ and $W^{(l+1)}$ do not commute, nor do activations and linear maps. Two computational graphs with the same edge set but different order can represent different functions, while homology/cohomology assigns them the same classes. That is:

> Cohomological information is a **quotient** of the cochain space (modulo the exact part), not an equivalent complete description. Compressing via cohomology discards order.

Consequences:

- using cohomology classes as the weight representation requires extra order encoding (directed complexes, path complexes, operads, or higher-categorical structures; Grigor'yan et al., 2014; May, 1972);
- an order-preserving representation is generally larger than the abelian quotient, so **matching the same precision may require more parameters** than the homology compression estimate;
- the decay assumption in Section 5 applies to weights after homological compression and does not directly imply that the original network parameters compress.

## 5. Hilbert Space and the Decay Assumption

### 5.1 Representation Space

Take the direct sum of cochain spaces of all orders:

$$
\mathcal{H} = \bigoplus_k C^k,\qquad \langle w, w'\rangle = \sum_k \langle w_k, w'_k\rangle_k,\qquad \|w\|^2 = \sum_k \|w_k\|^2
$$

In finite dimension it is automatically a Hilbert space. The motivation for this inner product is that orthogonality, projection, and truncation error only make sense once an inner product is fixed.

### 5.2 Decay Assumption

**Assumption (order decay)**: there are constants $C>0$ and $\rho \in (0,1)$ such that

$$
\|w_k\| \le C \rho^k
$$

(or polynomial decay $\|w_k\| \le C k^{-\alpha}$).

If the function depends on the weights in a Lipschitz way (constant $L$), the error of truncating at order $K$ satisfies

$$
\|f_w - f_{w_{\le K}}\| \le L \sum_{k>K} \|w_k\| \le \frac{L C \rho^{K+1}}{1-\rho}
$$

so the error decreases exponentially with $K$: **when the assumption holds**, low-order weights suffice for a high-precision representation. This is the testable form of the conjecture in Section 1.

The bound relies on two premises: the function is Lipschitz in the weights (finite constant $L$), and the representation space is regular enough (5.4); if $L$ is uncontrollable or the space is incomplete, the exponential bound fails.

### 5.3 Why It Might Hold

- high-order interactions of smooth target functions are usually weak;
- weight decay and similar regularizers suppress high-order terms;
- network training shows spectral bias: low-frequency/low-order structure is learned first (Rahaman et al., 2019);
- the low-dimensional structure of the data makes high-order combinations redundant.

These are intuitions, not proofs, and they do not rule out complementary, incompressible high-order information (5.5); Section 7 gives the test plan.

### 5.4 Regularity Assumptions on the Representation Space

A distinction matters here: truncated to finite order, the space is finite-dimensional and automatically complete; this section discusses the infinite-order limit, or embedding the weights into a function space (e.g. a Sobolev space).

The derivation above assumes $\mathcal{H}$ is a "nice" space: complete, bounded, separable, with a well-behaved norm. This need not hold in practice:

- **Completeness**: only complete inner-product spaces are Hilbert spaces; if we take a direct sum of finitely many cochain spaces and then a limit, we must complete it first, or Cauchy sequences may have no limit and the truncation bound loses meaning;
- **Boundedness**: weight spaces are usually unbounded (norms can be arbitrarily large); convergence and generalization require extra constraints (weight decay, spectral-norm bounds, compactness assumptions);
- **Separability and nuclearity**: kernel or Gaussian-process interpretations need additional structure such as separability or nuclearity;
- **Norm choice**: weight norm and function error differ by a Lipschitz constant; if that constant is unbounded or uncontrollable, the exponential bound is not tight;
- **Other exotic properties**: non-local, non-commutative, or singular spaces can break projection and orthogonal decomposition.

Practical use therefore needs **additional assumptions**: completion, bounded or compact domains, controllable Lipschitz constants, a separable basis, and a good norm (e.g. an embedding into a Sobolev norm). When these fail, the decay conjecture remains heuristic.

### 5.5 Complementarity: Low Order Does Not Necessarily Compress High Order

The Hodge decomposition already gives a structural reason: harmonic representatives in cohomology classes $H^k$ cannot be generated from lower-order potentials $\delta u$. If such weights do not decay with order but carry information that low-order terms cannot express, then:

- low-order parameters cannot compress them, and forcible truncation directly loses accuracy;
- high- and low-order information are **complementary**, not one containing the other;
- the decay assumption describes only tasks with "high-order redundancy" and cannot be generalized to a universal fact.

An intuitive example: parity-like functions require high-order interactions (O'Donnell, 2014); no amount of low-order terms can represent them. Real tasks usually lie in between — some high-order information is redundant (compressible), some is irreducible (must be kept).

The correct statement is therefore: **the compressibility of high-order weights is an empirical question that depends on the task and the representation space; one cannot assume a priori that low order suffices to represent high order.**

## 6. Implementation Paths

### 6.1 Explicit Higher-Order Layers

Add low-rank higher-order terms to a standard MLP:

$$
f(x) = W_1 x + \sum_{r} a_r (u_r^\top x)(v_r^\top x) + \sum_{r} b_r (p_r^\top x)(q_r^\top x)(s_r^\top x) + \cdots
$$

Use CP/Tucker decompositions to control parameter count and avoid combinatorial blow-up (Kolda & Bader, 2009).

### 6.2 Complex Parameterization

Learn cochains directly on the complex: parameterize the lower-order potential $u$, the harmonic representative $h$, and a higher-order potential $v$ separately, and reconstruct $w = \delta u + h + \delta^{*} v$ by Hodge decomposition in the forward pass. High-order weights are then generated from these potentials rather than stored independently.

### 6.3 Post-hoc Decomposition

Decompose a trained MLP (Taylor / ANOVA / Sobol), estimate each order's contribution, measure norm decay, then truncate high orders and fine-tune.

### 6.4 Regularization

Add $\lambda \sum_k k^{p} \|w_k\|^2$ to the loss to explicitly encourage high-order decay.

## 7. Test Plan

- **P0 order spectrum**: on small MLPs (polynomials, Boolean functions, images, regression), estimate how $\|w_k\|$ varies with $k$ using ANOVA / Sobol-type methods (Hoeffding, 1948; Sobol, 1993);
- **P1 truncation**: fix $K$, measure accuracy versus $K$, and check for exponential decay;
- **P2 compression ratio**: at equal accuracy, compare the parameter count of the complex representation with a standard MLP;
- **P3 counterexamples**: parity, high-order Boolean functions, and other intrinsically high-order tasks to test whether the decay assumption fails;
- **P4 regularization**: how order penalties trade off accuracy against decay;
- **P5 structure comparison**: order spectra and low-order retention for a plain MLP, a residual MLP, and a U-Net-style MLP.

Success criterion: on at least one task, a small $K$ (e.g. $K \le 3$) matches the accuracy of a full-order MLP with significantly fewer parameters.

## 8. Difficulties and Open Questions

- **Order is not uniquely defined**: Taylor expansion depends on the expansion point; ANOVA depends on the data distribution; complex order depends on vertex choice. A unified definition is needed;
- **Computational graph versus interaction graph**: $W$ connects inter-layer units, while interaction terms connect input coordinates; how do they coexist on the same complex?
- **Combinatorial explosion**: the number of $k$-simplices over $n$ vertices is $\binom{n}{k+1}$; high-order terms must be sparse or low-rank;
- **Interpretability of homology**: what function properties do $H^k$ classes correspond to? Currently this is only a formal analogy;
- **The decay assumption may fail**: some tasks are intrinsically high-order, and high/low-order information may be complementary, so low order cannot compress high order;
- **Choice of norm**: the Lipschitz constant between weight norm and function error may be large, so the exponential bound need not be tight;
- **Order versus the abelian assumption**: homology quotients out computation order; how can directed complexes / path complexes / operads retain it, and how many extra parameters does an order-preserving representation cost?
- **Regularity of the representation space**: how do completion, boundedness, the Lipschitz constant, and the norm choice affect truncation error and generalization?
- **Implicit order and residuals**: how can the order implicit in depth be measured, and what structure do residual/skip connections correspond to in homology?

## 9. Related Work

- Function decomposition: ANOVA, Sobol sensitivity, Taylor expansion (Hoeffding, 1948; Sobol, 1993; Hooker, 2004);
- Higher-order interaction networks: $\Pi$-Net, bilinear pooling, tensor decompositions (Jayakumar et al., 2020; Lin et al., 2015; Kolda & Bader, 2009);
- Topological deep learning: simplicial networks, cellular complexes, the Hodge Laplacian (Carlsson, 2009; Ebli et al., 2020; Bodnar et al., 2021; Lim, 2020);
- Spectral bias and the NTK (first-order approximation) (Rahaman et al., 2019; Jacot et al., 2018);
- Low-rank compression, pruning, and distillation.

## 10. Roadmap

| Version | Goal | Criterion |
| --- | --- | --- |
| v0.1 | Order-spectrum tooling | stable estimates of per-order norms on small networks |
| v0.2 | Truncation experiments | error decays exponentially with $K$ on at least one task |
| v0.3 | Complex parameterization prototype | fewer parameters than a standard MLP at equal accuracy |
| v1.0 | Full report | positive and negative results, ablations, reproducible |

## 11. TODO

- [ ] Fix a unified mathematical definition of "order"
- [ ] Implement Taylor / ANOVA order-spectrum estimation
- [ ] Measure decay on small networks
- [ ] Design a complex-parameterization prototype
- [ ] Estimate the parameter cost of preserving order
- [ ] Pin down regularity assumptions on the representation space
- [ ] Find intrinsically high-order tasks as counterexamples

## 12. Glossary

| Term | Meaning |
| --- | --- |
| Simplicial complex | a set of simplices closed under taking subsets |
| $k$-simplex | a combination of $k+1$ vertices |
| Chain complex | a graded vector space with boundary operators $\partial$ |
| Cochain | a weight functional defined on simplices |
| Homology / cohomology | invariants describing "holes" and irreducible weight patterns |
| Hodge decomposition | splitting cochains into exact, harmonic, and coexact parts |

## 13. References

1. Hatcher, A. (2002). *Algebraic Topology*. Cambridge University Press.
2. Lim, L.-H. (2020). Hodge Laplacians on Graphs. *SIAM Review*, 62(3).
3. Carlsson, G. (2009). Topology and Data. *Bulletin of the American Mathematical Society*, 46(2).
4. Ebli, S., Defferrard, M., & Spreemann, G. (2020). Simplicial Neural Networks. *NeurIPS 2020 Workshop on Topological Data Analysis and Beyond*.
5. Bodnar, C., Frasca, F., Otter, N., Wang, Y., Liò, P., Montúfar, G., & Bronstein, M. (2021). Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks. *ICML 2021*.
6. Grigor'yan, A., Lin, Y., Muranov, Y., & Yau, S.-T. (2014). Homologies of path complexes and digraphs. arXiv:1207.2834.
7. May, J. P. (1972). *The Geometry of Iterated Loop Spaces*. Springer.
8. Hoeffding, W. (1948). A Class of Statistics with Asymptotically Normal Distribution. *Annals of Mathematical Statistics*, 19(3).
9. Sobol', I. M. (1993). Sensitivity Estimates for Nonlinear Mathematical Models. *Mathematical Modelling and Computational Experiments*, 1(4).
10. Hooker, G. (2004). Discovering Additive Structure in Black Box Functions. *KDD 2004*.
11. Jayakumar, S. M., Czarnecki, W. M., Menick, J., Schwarzschild, A., Rae, J., Osindero, S., Teh, Y. W., Turner, R., & Pascanu, R. (2020). Multiplicative Interactions and Where to Find Them. *ICLR 2020*.
12. Lin, T.-Y., RoyChowdhury, A., & Maji, S. (2015). Bilinear CNN Models for Fine-grained Visual Recognition. *ICCV 2015*.
13. Kolda, T. G., & Bader, B. W. (2009). Tensor Decompositions and Applications. *SIAM Review*, 51(3).
14. Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural Tangent Kernel: Convergence and Generalization in Neural Networks. *NeurIPS 2018*.
15. Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F. A., Bengio, Y., & Courville, A. (2019). On the Spectral Bias of Neural Networks. *ICML 2019*.
16. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
17. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*.
18. O'Donnell, R. (2014). *Analysis of Boolean Functions*. Cambridge University Press.

## 14. Changelog

| Date | Change |
| --- | --- |
| 2026-11-28 | First draft: the full chain graph → complex → homology → Hilbert decay assumption, with a test plan |
| 2026-11-28 | v1.1: added the gap between ordered computation and abelian homology (4.3) and regularity assumptions such as completeness/boundedness of the representation space (5.4); restated the compression conclusion as an optimistic upper bound |
| 2026-11-28 | v1.2: added 3.3 on implicit higher-order information in depth and low-order loss, explaining how residual/U-Net structures preserve low-order information; added the P5 structure-comparison experiment |
| 2026-11-28 | v1.3: corrected the claim — high- and low-order information may be complementary, and low-order compression of high-order weights is only a conditional hypothesis; added 5.5 |
| 2026-11-28 | Review: corrected the Hodge decomposition to "exact ⊕ harmonic ⊕ coexact", clarified the finite/infinite-order completeness distinction, and added references |
