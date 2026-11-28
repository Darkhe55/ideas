# Transformer 的隐式信息表示（扩展中途版）

> 阶段：**扩展中途版**（约 8 页） · 状态：**建设中** · 最后更新：2026-11-28
> 语言：中文 · [English](../en/extended/implicit-information-representation-in-transformers.md)

## 1. 观察：结构相同，语义分叉

看三个句子：

- 我是猫。
- 我是医生。
- 我是司机。

它们的句法结构完全一致（“我是 + 名词”的模板），但语义上后两者明显更近：医生与司机都是**职业**，猫是**物种**。

如果 Transformer 的 KV 投影真的编码了语义结构，那么可以猜测：

- “我是医生”与“我是司机”在 KV 投影下的**形状**更相似；
- “我是猫”与它们只保持部分相似（结构像、语义远）；
- 毫不相干的句子不共享这种形状。

这里的“形状”不是绝对坐标，而是句子表示在 KV 空间中的**相对构型**：token 向量之间的夹角、距离、子空间结构。它应当在旋转/平移下不变。

## 2. KV 投影与“形状”

### 2.1 KV 投影

Transformer（Vaswani et al., 2017）每一层把隐状态 $X$ 投影为 key 与 value：

$$
K = X W_K,\qquad V = X W_V
$$

一个句子因此得到一组 key 向量 $\{k_t\}$ 和一组 value 向量 $\{v_t\}$。注意力由这些向量的内积决定，输出由 value 的加权和决定。

### 2.2 形状的度量

把句子的表示看成点云，形状可以用中心化 Gram 矩阵描述：

$$
G_{ij} = \big\langle k_i - \bar{k},\; k_j - \bar{k} \big\rangle
$$

两个形状之间的距离可以用：

- **Procrustes 距离**：先做正交对齐，再比较 Frobenius 范数；
- **子空间主角度**：比较两个点云张成子空间的夹角；
- **CKA**：比较表示之间的线性/核相似性。

这些度量都对整体旋转与平移不敏感，符合“形状”的直觉。

### 2.3 猜测

- **G1**：语义越近，KV 形状距离越小；
- **G2**：结构相同但语义不同，形状只部分相似；
- **G3**：无关句子之间没有稳定的共同形状。

这些猜测可以用模板句直接检验（第 6 节）。

## 3. 无先验嵌入的唯一性：正则单形

### 3.1 无先验假设

假设词嵌入到欧氏空间，并且**没有任何先验**：所有词在对称意义下等价，不存在“某些词天然更近”这样的设定。

更精确地说，“无先验”指词集上的任意置换都可由空间等距实现（嵌入具有完全对称群）；定理 1 将说明，这唯一地刻画了正则单形。

### 3.2 定理与证明

**定理 1**（$n \ge 2$）：若 $n$ 个词两两等距，则它们必然是正则 $(n-1)$-单形的顶点；该嵌入在等距变换下唯一。

**证明**：设嵌入为 $v_1, \dots, v_n \in \mathbb{R}^d$，所有词对距离为常数 $d_0$，质心 $\bar{v} = \frac{1}{n}\sum_i v_i$。

对任意 $i$，有恒等式

$$
\sum_{j=1}^{n} \|v_i - v_j\|^2 = n\|v_i - \bar{v}\|^2 + \sum_j \|v_j - \bar{v}\|^2
$$

由于所有距离相等，左边为 $(n-1)d_0^2$，右边两项各自对所有 $i$ 相等，因此 $\|v_i - \bar{v}\|^2$ 为常数：

$$
\|v_i - \bar{v}\|^2 = \frac{(n-1)d_0^2}{2n}
$$

于是对 $i \ne j$，

$$
\langle v_i - \bar{v},\, v_j - \bar{v}\rangle
= \frac{s_i + s_j - d_0^2}{2}
= -\frac{d_0^2}{2n}
$$

中心化 Gram 矩阵 $G$ 因此形如

$$
G = \frac{d_0^2}{2} I - \frac{d_0^2}{2n} J
$$

其中 $J$ 是全 1 矩阵。它的特征值是 $d_0^2/2$（重数 $n-1$）与 $0$（重数 1），秩为 $n-1$。所以这些点位于一个 $(n-1)$ 维仿射子空间，且 Gram 矩阵恰好是正则单形的 Gram 矩阵；由 Gram 矩阵决定点集在等距意义下的唯一性，得证。$\square$

### 3.3 推论

- 若嵌入不是等距的，就存在两个词的距离小于另外两个词的距离——这等价于引入了一个**先验**；
- 所以在“无先验”的约束下，$n$ 个词的嵌入只有一种取法（至多差一个等距变换）：正则单形；
- 单形的嵌入维度是 $n-1$。想降维，就必须引入先验（哪些词应当更近、哪些维度可以丢弃）。

## 4. 单形嵌入下的投影族

### 4.1 设定

取标准单形的顶点 $e_1, \dots, e_n$（$\mathbb{R}^n$ 的标准基）。顶点子集 $S \subseteq \{1, \dots, n\}$ 张成一个子单形

$$
\operatorname{aff}(S) = \Big\{ \sum_{i \in S} \lambda_i e_i \;:\; \sum_i \lambda_i = 1 \Big\}
$$

其维数为 $|S| - 1$。

### 4.2 命题

**命题 2**：设整数 $k \ge 1$。对任意两个顶点子集 $S, T$，只要 $|S|, |T| > k+1$，就存在一族低维投影 $\mathcal{F}(S, T)$（投影到 $\mathbb{R}^k$），使得：

1. 对任意 $P \in \mathcal{F}(S, T)$，都有 $P(\operatorname{aff}(S)) = P(\operatorname{aff}(T))$；
2. 对任意顶点子集 $U$，$|U| > k+1$，只要 $S \nsubseteq U$ 且 $T \nsubseteq U$，就存在 $P \in \mathcal{F}(S, T)$ 使得 $P(\operatorname{aff}(U)) \ne P(\operatorname{aff}(S))$。

> 说明：条件 $|S| > k+1$ 保证投影到 $\mathbb{R}^k$ 时不超过子单形的维数，同时给构造留出足够顶点。这里的“投影”指到低维空间的线性映射，不要求是幂等投影。排除 $S \subseteq U$ 或 $T \subseteq U$ 是必要的：这两种情况下 $P(\operatorname{aff}(U)) \supseteq P(\operatorname{aff}(S))$，而所有像都落在 $\mathbb{R}^k$ 里，像必然相等（见 4.4）。

### 4.3 构造与证明

取 $\mathbb{R}^k$ 中的正则单形 $Q = \{q_0, \dots, q_k\}$（$k+1$ 个顶点）。对任意一对满射

$$
\pi_S: S \to Q,\qquad \pi_T: T \to Q
$$

定义线性映射 $P$：在标准基上令

$$
P(e_s) = q_{\pi_S(s)}\ (s \in S),\qquad
P(e_t) = q_{\pi_T(t)}\ (t \in T),
$$

对 $S \cup T$ 之外的顶点任意取值（例如都映到某个 $q$）。线性延拓到整个 $\mathbb{R}^n$。所有这样的 $P$ 构成族 $\mathcal{F}(S, T)$。

**（1）像重合**：$P(\operatorname{aff}(S))$ 是 $\{q_0, \dots, q_k\}$ 的凸包，即 $\operatorname{aff}(Q)$；同理 $P(\operatorname{aff}(T)) = \operatorname{aff}(Q)$。所以两者像相等。

**（2）区分 $U$**：因为 $S \nsubseteq U$，取 $s^* \in S \setminus U$；因为 $T \nsubseteq U$，取 $t^* \in T \setminus U$。固定一个 $q^* \in Q$，令

$$
\pi_S(s^*) = \pi_T(t^*) = q^*,
$$

并让 $S$ 中其余元素、$T$ 中其余元素、以及 $S \cup T$ 外的顶点都映到 $Q \setminus \{q^*\}$。这是可行的满射：$q^*$ 已有原像 $s^*, t^*$，其余 $k$ 个顶点由 $S \setminus \{s^*\}$（至少 $k$ 个元素）覆盖。

于是 $U$ 中来自 $S$ 或 $T$ 的顶点都不会映到 $q^*$（它们不是 $s^*, t^*$），$U$ 的其余顶点也被安排避开 $q^*$。因此 $P(\operatorname{aff}(U))$ 不包含 $q^*$，而 $P(\operatorname{aff}(S)) = \operatorname{aff}(Q)$ 包含 $q^*$，两者不等。$\square$

### 4.4 退化情形：包含关系在投影下不可见

若 $S \subseteq U$，则 $\operatorname{aff}(U) \supseteq \operatorname{aff}(S)$，于是对任意线性 $P$ 有 $P(\operatorname{aff}(U)) \supseteq P(\operatorname{aff}(S))$；而所有像都落在 $\mathbb{R}^k = \operatorname{aff}(Q)$ 内，所以 $P(\operatorname{aff}(U)) = \operatorname{aff}(Q) = P(\operatorname{aff}(S))$。也就是说，**包含关系不可能被这类投影区分**。

这不是证明的缺陷，而是单形对称性的直接体现：它说明单形嵌入本身不携带任何“哪些子集包含哪些子集”之外的语义区分。

### 4.5 含义

- 在正则单形里，任意两个子单形都能被一族投影压成同一形状；同一族里又存在投影区分第三个（只要它不包含前两个）；
- 单形的对称性极强：投影能“抹平”任意两个子集，说明**形状相似性本身不构成语义区分**；
- 因此 Transformer 中观察到的“职业形状 vs 物种形状”，只能是训练把对称的单形几何塑造成的结果，而不是嵌入几何自动给出的。

## 5. 对 Transformer 隐式表示的启示

### 5.1 猜测 G4：KV 形状是隐式的语义表示

训练会把无先验的单形几何逐渐塑造成有层次的形状：职业类句子在 KV 空间里形成相近的低维构型，物种类形成另一簇。这与第 1 节的观察一致。

### 5.2 可能的训练方法

1. **形状对比学习**：同一语义类的句子对拉近 Procrustes 距离，不同类推远；
2. **层级语义监督**：用“物种 / 职业 / 其他”这样的标签作为先验，指导嵌入或 KV 投影；
3. **单形初始化**：用正则单形初始化词嵌入（等距、无先验），再让训练逐步破坏对称性；
4. **降维**：在保持形状区分度（如语义三元组准确率）的前提下降低 KV 维度，用性能决定维数；
5. **形状正则**：对 KV 的 Gram 矩阵加约束，鼓励类内稳定、类间分离。

### 5.3 注意事项

- KV 形状受层数、注意力头、位置编码影响，句子级形状需要先定义聚合方式；
- 形状相似不等于因果贡献，需要干预实验；
- 语义标签本身是人为先验，与“无先验”的理想并不一致。

## 6. 可检验预测与实验

- **E1 模板句**：对“我是 X”生成一组句子，测 KV 形状距离矩阵，检查职业 vs 物种是否分簇；
- **E2 层间对比**：浅层可能更偏结构，深层更偏语义；
- **E3 随机 vs 训练**：随机初始化的形状是否更接近单形（等距）？
- **E4 单形初始化**：与标准初始化对比，训练后的语义分簇是否更好/更差；
- **E5 降维曲线**：把 KV 投影到 $d'$ 维，测形状分类准确率随 $d'$ 的变化；
- **E6 形状训练**：加入形状对比损失后，距离矩阵是否按语义层级变化。

## 7. 局限

- **单形假设过强**：真实词嵌入受数据、训练、优化影响，远离等距；
- **零模型的边界**：单形是最小假设的零模型；真实 Transformer 有位置编码、层归一化、训练与数据偏置，KV 几何会显著偏离单形。命题 2 是几何直觉的形式化，只说明“形状相似本身不足以承载语义区分”，不是关于真实 KV 形状的断言；
- **形状定义不唯一**：Gram、子空间、Procrustes 各有偏差；
- **聚合方式**：句子级形状依赖于 token 如何汇总（均值、注意力池化等）；
- **混淆因素**：位置编码、层归一化、残差都会改变几何；
- **因果性**：形状相似只是相关，不能直接说明模型“按形状推理”。

## 8. 相关工作

- 词嵌入几何：各向异性、锥形分布（Ethayarajh, 2019; Gao et al., 2019）；
- 表示相似性：CKA、Procrustes 对齐、子空间方法（Kornblith et al., 2019; Schönemann, 1966）；
- 等距码与紧框架：正则单形、等角紧框架（Strohmer & Heath, 2003; Sustik et al., 2007）；
- 叠加表示与线性表示假设（Elhage et al., 2022; Park et al., 2023）；
- 几何深度学习中的形状与对称性（Bronstein et al., 2021）。

## 9. 路线图

| 版本 | 目标 | 判定标准 |
| --- | --- | --- |
| v0.1 | 形状度量工具 | Procrustes / 子空间距离可用 |
| v0.2 | 模板句实验 | 职业 vs 物种形状可分 |
| v0.3 | 命题 2 完整化 | 覆盖一般 k 与所有退化情形 |
| v0.4 | 形状对比训练 | 距离矩阵按语义层级变化 |
| v1.0 | 完整报告 | 正反例、消融、可复现 |

## 10. TODO

- [ ] 定义句子级 KV 形状的标准聚合方式
- [ ] 在开源模型上测“我是 X”形状距离
- [ ] 验证随机初始化是否接近等距
- [ ] 实现形状对比损失
- [ ] 把命题 2 推广到更一般的复形

## 11. 术语

| 术语 | 含义 |
| --- | --- |
| KV 投影 | 注意力中对 key/value 的线性映射 |
| 形状 | 表示点云在旋转/平移下不变的几何构型 |
| 正则单形 | 所有顶点两两等距的单纯形 |
| Procrustes 距离 | 正交对齐后的最小 Frobenius 距离 |
| 子空间主角度 | 两个子空间之间的夹角序列 |

## 12. 参考文献

1. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *NeurIPS 2013*.
3. Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. *EMNLP 2019*.
4. Gao, J., He, D., Tan, X., Qin, T., Wang, L., & Liu, T.-Y. (2019). Representation Degeneration Problem in Training Natural Language Generation Models. *ICLR 2019*.
5. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *ICML 2019*.
6. Schönemann, P. H. (1966). A Generalized Solution of the Orthogonal Procrustes Problem. *Psychometrika*, 31(1).
7. Strohmer, T., & Heath, R. W. (2003). Grassmannian Frames with Applications to Coding and Communication. *Applied and Computational Harmonic Analysis*, 14(3).
8. Sustik, M. A., Tropp, J. A., Dhillon, I. S., & Heath, R. W. (2007). On the Existence of Equiangular Tight Frames. *Linear Algebra and its Applications*, 426(2–3).
9. Elhage, N., Hume, T., Olsson, C., et al. (2022). Toy Models of Superposition. *Transformer Circuits Thread*.
10. Park, K., Choe, Y. J., & Veitch, V. (2023). The Linear Representation Hypothesis and the Geometry of Large Language Models. arXiv:2311.03658.
11. Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. arXiv:2104.13478.
12. Hewitt, J., & Manning, C. D. (2019). A Structural Probe for Finding Syntax in Word Representations. *NAACL 2019*.

## 13. 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-11-28 | 初稿：形状观察、单形唯一性、投影族命题与 Transformer 启示 |
| 2026-11-28 | 校订：把“无先验”形式化为置换对称；补充“投影”与零模型边界说明；补引表示几何与编码理论文献 |
