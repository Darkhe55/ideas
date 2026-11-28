# 没实现自驱力时的模型伪 RSI 方法（扩展中途版）

> 阶段：**扩展中途版**（约 7 页） · 状态：**建设中** · 最后更新：2026-11-28
> 语言：中文 · [English](../en/extended/pseudo-rsi-without-self-drive.md)

## 1. 问题与定位

**RSI**（Recursive Self-Improvement，递归自我改进）指模型在循环中改进自身能力：改进后的模型再改进下一轮，形成正反馈。真正的 RSI 通常隐含**自驱力**：模型自己提出目标、自己安排课程、自己决定何时继续。

当前模型还谈不上稳定的自驱力：目标由人给，课程由人排，停止由人判断。于是可以退一步，做一种**伪 RSI**：

> 标准由人定，判断与训练由模型做，人只在关键节点介入。

它不是"模型自己决定要变强"，而是"人定好靶子，模型自己瞄准、自己练习、自己评估"。

需要先划清界限：**伪 RSI 优化的是模型能力本身，不是某一次产物的质量**。自我审查、重写、best-of-N 这类做法只是推理时的搜索，不会改变参数，也不会让下一轮更强；它们属于"产物优化"，不是 RSI。

## 2. 核心思路

一个伪 RSI 循环由四件事组成：

1. **标准** $S$：人给出的、可检查的质量与能力要求；
2. **执行** $E_\theta$：当前模型产出答案、代码、数据或语料；
3. **判断** $J_\phi$：按 $S$ 评价产物、排序、筛选；
4. **训练**：用被判断筛选过的数据更新模型参数 $\theta$。

```
标准 S
  │
  ▼
执行 E_θ  ──►  产物 a
  │                │
  │                ▼
  │            判断 J_φ（按 S 打分/排序/筛选）
  │                │
  │                ▼
  └──────────  自我收集的数据 D_t
                   │
                   ▼
              训练 θ_{t+1}
                   │
                   ▼
              回到执行/判断
```

与"人标注所有数据"相比，人的劳动从**逐条标注**转移到**定标准、审判断、补盲区**。模型自行收集语料（检索、生成、筛选），人只抽查。

## 3. 前提：判断能力与执行能力必须有分别

伪 RSI 的发动机是"用判断提高执行"。这要求判断不是执行的复读机：

- 如果 $J$ 与 $E$ 共享同一套偏差，模型会把自己的错误判成正确，循环放大错误；
- 如果 $J$ 只是 $E$ 的又一次采样，它没有额外信息，筛选等价于随机。

形式化地，设执行器 $E_\theta: \mathcal{X} \to \mathcal{A}$，判断器 $J_\phi: \mathcal{X} \times \mathcal{A} \to [0,1]$。给定人类偏好对 $(a^+, a^-)$，定义判断的**区分度**

$$
\delta(J) = \Pr\big[J(x, a^+) > J(x, a^-)\big] - \tfrac{1}{2}
$$

前提是 $\delta(J) > 0$ 且足够大（例如高于随机但不必完美），并且 $J$ 的错误与 $E$ 的错误**不能完全相关**。否则自训练没有信息增量。

实践中制造"分别"的手段：

- 判断器与执行器不同源：不同提示、不同模型、集成投票、外部工具校验；
- 判断器更小、更冻结，避免随训练一起漂移；
- 用人类标注的锚点集校准判断器（可靠性、AUC、校准误差）；
- 判断只做**相对排序**（A 比 B 好），不做绝对打分，降低尺度漂移。

## 4. 方法：以标准为目标的迭代

### 4.1 标准从哪来

标准应当显式、可检查、可版本化：

- 能用程序检查的（单元测试、格式、引用、约束）优先机器检查；
- 不能机器检查的（有用性、风格、安全性）写成 rubric，由判断器执行、人类抽查；
- 标准要冻结版本；标准一改，前后两轮的能力不可比。

### 4.2 三个子循环

| 循环 | 做什么 | 人的角色 |
| --- | --- | --- |
| 数据循环 | 模型检索/生成/筛选语料，构造训练集 | 抽查来源与质量 |
| 训练循环 | 在筛选数据上微调，验证集上评估 | 定算力预算、看曲线 |
| 判断循环 | 用人类锚点校准判断器 | 标注锚点、审计分歧 |

三个循环的瓶颈通常是**判断循环**：判断器不够准，数据循环和训练循环都会放大噪声。

### 4.3 自我收集语料

"模型自行收集"不等于"模型自己编答案"：

- 从文档、代码库、工具输出中检索原始材料；
- 用判断器筛选与标准相关的片段；
- 生成问答/改写/推理链，再用判断器过滤；
- 保留来源与许可证信息，过滤不安全或侵权内容；
- 与人写数据按比例混合，防止分布收窄。

### 4.4 防退化

- **锚点集**：一小份人工标注的高质量数据，每轮都测，防止自评漂移；
- **多样性**：多来源、多任务、多语言混合，监控覆盖率；
- **多判断器**：至少两个独立判断器，分歧大的样本交人审；
- **对抗检查**：专门构造"看起来对但错"的样本，测判断器是否被骗。

## 5. 与"自我审查产物"的区别

| 维度 | 自我审查产物（self-refine） | 伪 RSI |
| --- | --- | --- |
| 优化对象 | 单次产物 | 模型能力（参数/数据/判断器） |
| 成本形态 | 每次推理重复消耗 | 一次训练、长期摊销 |
| 是否变强 | 否，模型不变 | 是，下一轮更强 |
| 判断要求 | 只要能改好这一份 | 要能跨样本稳定区分 |
| 主要风险 | 搜索轨迹偏差 | 判断器漂移、数据坍缩 |

用户指出的关键现象是：**优化产物的过程本身是一个动力学过程，轨迹会干扰判断**。第 $t$ 次迭代时，判断器看到的上下文、已经形成的惯性、对"再改一次是否值得"的成本感，都会改变它的评分：

$$
J_t = J(\cdot \mid \tau_t),\qquad \tau_t = (a_1, a_2, \dots, a_t)
$$

于是可能出现：产物其实没达标，但模型在轨迹 $\tau_t$ 上判断"已经改得够多、继续改是浪费"，于是提前停止。这不是能力不足，而是**判断被优化过程污染**。

因此伪 RSI 要额外做三件事：

1. **冻结判断器**：判断器不参与被优化的轨迹，或用独立会话、重置上下文；
2. **预注册停止规则**：停止条件在看结果之前定好，不随轨迹调整；
3. **外部评估**：用锚点集或人工审计判定是否真的达标，而不是相信模型自己的"够了"。

## 6. 缺陷与边界

- **不是完全自动化**：标准要人定，判断器要人校准，盲区要人补；模型只是把"标注"换成了"筛选"，把"执行"换成了"自评"；
- **判断天花板**：判断器的区分度决定循环上限；判断器弱，自训练会把噪声固化；
- **数据坍缩**：只用模型自生成数据，分布会收窄（model collapse）；
- **标准被游戏化**：模型可能学会迎合 rubric 的表面特征（Goodhart）；
- **停止偏差**：轨迹污染导致的提前停止或过度迭代；
- **安全与合规**：自行收集的语料可能含侵权、隐私、有害内容，需要过滤与审计。

但相对"全部数据人工标注"，人力确实省下了一大块：人从"生产者"变成"审计者"。

## 7. 评测

| 维度 | 指标 | 说明 |
| --- | --- | --- |
| 判断区分度 | AUC、与人类偏好一致率、校准误差 | 前提是否成立 |
| 能力增益 | 每轮在留出基准上的提升 | 是真变强还是过拟合 |
| 人力节省 | 人工标注量 / 审计量之比 | 核心卖点 |
| 数据质量 | 自收集数据的来源多样性、过滤率 | 防坍缩 |
| 停止正确性 | 预注册停止 vs 模型自判停止的差异 | 检验轨迹污染 |
| 安全性 | 有害内容比例、许可证合规率 | 上线门槛 |

关键实验：

- **E1**：判断器与执行器同源 vs 异源，比较自训练增益；
- **E2**：冻结判断器 vs 随训练更新判断器，比较退化速度；
- **E3**：预注册停止 vs 模型自判停止，检测提前停止率；
- **E4**：纯自收集数据 vs 混合人工数据，比较分布收窄；
- **E5**：多轮循环后能力是否持续上升，还是进入平台期。

## 8. 相关工作

- RSI 与种子 AI：Good (1965)、Yudkowsky (2008)；
- Gödel machine：Schmidhuber (2003)，要求形式化证明，比伪 RSI 强得多也难得多；
- 自训练与拒绝采样：STaR（Zelikman et al., 2022）、拒绝采样微调；
- 自我改进与自我奖励：Self-Rewarding LMs（Yuan et al., 2024）、SPIN（Chen et al., 2024）；
- 自我精炼：Self-Refine（Madaan et al., 2023）；
- LLM 作为评判者：MT-Bench / LLM-as-a-Judge（Zheng et al., 2023）；
- 人类偏好与奖励模型：Christiano et al. (2017)、InstructGPT（Ouyang et al., 2022）；
- 生成对抗：GAN 的判别器/生成器（Goodfellow et al., 2014）；
- 模型坍缩：Shumailov et al. (2024)；
- 奖励黑客与 Goodhart：Amodei et al. (2016)、Skalse et al. (2022)；
- 弱到强泛化：Burns et al. (2023)。

## 9. 路线图

| 版本 | 目标 | 判定标准 |
| --- | --- | --- |
| v0.1 | 判断器校准 | 与人类偏好一致率显著高于随机 |
| v0.2 | 单轮自训练 | 留出基准上稳定提升 |
| v0.3 | 多轮 + 审计 | 判断器不漂移、能力持续上升 |
| v0.4 | 停止规则 | 预注册停止与人工判定一致 |
| v1.0 | 端到端报告 | 人力节省与能力增益可量化 |

## 10. TODO

- [ ] 定义可机器检查的标准模板
- [ ] 构造人类锚点集
- [ ] 实现判断器冻结/重置机制
- [ ] 设计预注册停止规则
- [ ] 测量多轮后的分布多样性

## 11. 术语

| 术语 | 含义 |
| --- | --- |
| 伪 RSI | 外部标准驱动、模型自评自训、人仍介入的改进循环 |
| 判断器 | 按标准评价产物的模型或程序 |
| 区分度 | 判断器把好坏样本排对的概率减去随机基线 |
| 轨迹污染 | 优化历史改变判断器的评分 |
| 锚点集 | 人工标注、用于校准与验收的固定样本 |

## 12. 参考文献

1. Good, I. J. (1965). Speculations Concerning the First Ultraintelligent Machine. *Advances in Computers*, 6.
2. Yudkowsky, E. (2008). Artificial Intelligence as a Positive and Negative Factor in Global Risk. In *Global Catastrophic Risks*.
3. Schmidhuber, J. (2003). Gödel Machines: Self-Referential Universal Problem Solvers Making Provably Optimal Self-Improvements. arXiv:cs/0309048.
4. Zelikman, E., Wu, Y., Mu, J., & Goodman, N. D. (2022). STaR: Bootstrapping Reasoning With Reasoning. *NeurIPS 2022*.
5. Yuan, W., Pang, R. Y., Cho, K., et al. (2024). Self-Rewarding Language Models. *ICML 2024*.
6. Chen, Z., Deng, Y., Yuan, H., Ji, K., & Gu, Q. (2024). Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models. *ICML 2024*.
7. Madaan, A., Tandon, N., Gupta, P., et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. *NeurIPS 2023*.
8. Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 Datasets and Benchmarks*.
9. Christiano, P. F., Leike, J., Brown, T. B., et al. (2017). Deep Reinforcement Learning from Human Preferences. *NeurIPS 2017*.
10. Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS 2022*.
11. Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al. (2014). Generative Adversarial Networks. *NeurIPS 2014*.
12. Shumailov, I., Shumaylov, Z., Zhao, Y., et al. (2024). AI Models Collapse When Trained on Recursively Generated Data. *Nature*, 631.
13. Amodei, D., Olah, C., Steinhardt, J., et al. (2016). Concrete Problems in AI Safety. arXiv:1606.06565.
14. Skalse, J., Howe, N. H. R., Krasheninnikov, D., & Krueger, D. (2022). Defining and Characterizing Reward Hacking. *NeurIPS 2022*.
15. Burns, C., Izmailov, P., Kirchner, J. H., et al. (2023). Weak-to-Strong Generalization: Eliciting Strong Capabilities with Weak Supervision. arXiv:2312.09390.

## 13. 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-11-28 | 初稿：伪 RSI 定位、判断/执行分离前提、以标准为目标的循环、轨迹污染与停止偏差、缺陷与评测 |
