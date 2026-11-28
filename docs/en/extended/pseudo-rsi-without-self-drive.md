# Pseudo-RSI Without Self-Drive (Extended)

> Stage: **In Progress** (~7 pages) · Status: **WIP** · Last updated: 2026-11-28
> Language: [中文](../../扩展中途版/没实现自驱力时的模型伪RSI方法.md) · English

## 1. Problem and Scope

**RSI** (Recursive Self-Improvement) means a model improves its own capability in a loop: the improved model improves the next round, forming positive feedback. Genuine RSI usually implies **self-drive**: the model proposes its own goals, designs its own curriculum, and decides when to continue.

Current models have no stable self-drive: goals are given, curricula are arranged, and stopping is decided by humans. So we take one step back and build a **pseudo-RSI**:

> Humans set the standards, the model judges and trains, and humans intervene only at key nodes.

It is not "the model decides to become stronger"; it is "humans set the target, and the model aims, practices, and evaluates itself."

A boundary must be drawn first: **pseudo-RSI optimizes the model's capability itself, not the quality of a single artifact**. Self-critique, rewriting, and best-of-N are inference-time search; they do not change parameters and do not make the next round stronger. They optimize artifacts, not capability — not RSI.

## 2. Core Idea

A pseudo-RSI loop has four parts:

1. **Standard** $S$: human-specified, checkable quality and capability requirements;
2. **Execution** $E_\theta$: the current model produces answers, code, data, or corpora;
3. **Judgment** $J_\phi$: scores, ranks, and filters artifacts according to $S$;
4. **Training**: updates parameters $\theta$ using the judged, filtered data.

```
standard S
  │
  ▼
execution E_θ  ──►  artifact a
  │                    │
  │                    ▼
  │              judgment J_φ (score/rank/filter by S)
  │                    │
  │                    ▼
  └────────────  self-collected data D_t
                       │
                       ▼
                  train θ_{t+1}
                       │
                       ▼
              back to execution/judgment
```

Compared with "humans label all data", human labor shifts from **per-item labeling** to **setting standards, auditing judgment, and filling blind spots**. The model collects corpora itself (retrieval, generation, filtering), and humans spot-check.

## 3. Precondition: Judgment Must Be Separate from Execution

The engine of pseudo-RSI is "raising execution with judgment". This requires judgment not to be a replay of execution:

- if $J$ shares the same biases as $E$, the model judges its own errors as correct and the loop amplifies them;
- if $J$ is just another sample of $E$, it carries no extra information and filtering is equivalent to randomness.

Formally, let the executor be $E_\theta: \mathcal{X} \to \mathcal{A}$ and the judge $J_\phi: \mathcal{X} \times \mathcal{A} \to [0,1]$. Given human preference pairs $(a^+, a^-)$, define the judge's **discrimination**

$$
\delta(J) = \Pr\big[J(x, a^+) > J(x, a^-)\big] - \tfrac{1}{2}
$$

The precondition is $\delta(J) > 0$ and large enough (above chance, not necessarily perfect), and the errors of $J$ must not be perfectly correlated with those of $E$. Otherwise self-training has no information gain.

In practice, separation can be created by:

- a judge from a different source than the executor: different prompts, different models, ensemble voting, external tool checks;
- a smaller, frozen judge that does not drift with training;
- human-labeled anchors to calibrate the judge (reliability, AUC, calibration error);
- **relative ranking** (A is better than B) rather than absolute scores, reducing scale drift.

## 4. Method: Iterating Toward a Standard

### 4.1 Where Standards Come From

Standards should be explicit, checkable, and versioned:

- whatever can be checked by programs (unit tests, formats, citations, constraints) should be checked by machines;
- whatever cannot (usefulness, style, safety) is written as a rubric, executed by the judge and spot-checked by humans;
- standards must be frozen by version; once they change, capability across rounds is not comparable.

### 4.2 Three Sub-loops

| Loop | What it does | Human role |
| --- | --- | --- |
| Data loop | the model retrieves/generates/filters corpora and builds training sets | spot-check sources and quality |
| Training loop | fine-tunes on filtered data, evaluates on validation | set compute budget, watch curves |
| Judgment loop | calibrates the judge with human anchors | label anchors, audit disagreements |

The bottleneck is usually the **judgment loop**: if the judge is inaccurate, the data and training loops amplify noise.

### 4.3 Self-Collected Corpora

"Self-collected" is not "the model makes up answers":

- retrieve raw material from documents, codebases, and tool outputs;
- filter segments relevant to the standard with the judge;
- generate QA/rewrites/reasoning chains, then filter with the judge;
- keep provenance and license information; filter unsafe or infringing content;
- mix with human-written data in a fixed ratio to prevent distribution narrowing.

### 4.4 Preventing Degeneration

- **Anchor set**: a small human-labeled, high-quality set measured every round to prevent self-evaluation drift;
- **Diversity**: mix sources, tasks, and languages; monitor coverage;
- **Multiple judges**: at least two independent judges; large disagreements go to human review;
- **Adversarial checks**: construct "looks right but wrong" samples to test whether judges can be fooled.

## 5. Difference from "Self-Reviewing Artifacts"

| Aspect | Self-refine | Pseudo-RSI |
| --- | --- | --- |
| Target | a single artifact | model capability (parameters/data/judge) |
| Cost shape | repeated inference per attempt | one training run, amortized |
| Does it get stronger | no, the model is unchanged | yes, the next round is stronger |
| Judgment requirement | just fix this one artifact | stable discrimination across samples |
| Main risk | search-trajectory bias | judge drift, data collapse |

The key phenomenon the user points to: **the artifact-optimization process is itself a dynamical process, and its trajectory interferes with judgment**. At iteration $t$, the judge's context, the momentum already built up, and the sense of whether "another pass is worth it" all change its scores:

$$
J_t = J(\cdot \mid \tau_t),\qquad \tau_t = (a_1, a_2, \dots, a_t)
$$

So it can happen that the artifact is below standard, yet the model judges on trajectory $\tau_t$ that "it has been revised enough and further work is wasteful" and stops early. This is not a capability deficit but **judgment contaminated by the optimization process**.

Pseudo-RSI therefore needs three extra safeguards:

1. **Freeze the judge**: the judge stays out of the optimized trajectory, or uses an independent session with reset context;
2. **Pre-register the stopping rule**: fix the stopping condition before seeing results; do not adapt it along the trajectory;
3. **External evaluation**: use anchors or human audit to decide whether the standard is really met, instead of trusting the model's own "enough".

## 6. Defects and Boundaries

- **Not fully automated**: humans set standards, calibrate judges, and fill blind spots; the model replaces "labeling" with "filtering" and "execution" with "self-evaluation";
- **Judgment ceiling**: the judge's discrimination bounds the loop; a weak judge freezes noise into the model;
- **Data collapse**: self-generated-only data narrows the distribution (model collapse);
- **Standard gaming**: the model may learn to satisfy surface features of the rubric (Goodhart);
- **Stopping bias**: early stops or over-iteration caused by trajectory contamination;
- **Safety and compliance**: self-collected corpora may contain infringing, private, or harmful material and need filtering and auditing.

But relative to "labeling all data by hand", a large block of labor is indeed saved: humans move from producers to auditors.

## 7. Evaluation

| Dimension | Metric | Description |
| --- | --- | --- |
| Judge discrimination | AUC, agreement with human preference, calibration error | whether the precondition holds |
| Capability gain | improvement per round on held-out benchmarks | real gain or overfitting |
| Labor savings | ratio of human labeling to auditing | the core selling point |
| Data quality | source diversity, filtering rate | prevents collapse |
| Stopping correctness | difference between pre-registered and model-decided stops | tests trajectory contamination |
| Safety | harmful content ratio, license compliance | deployment gate |

Key experiments:

- **E1**: same-source vs different-source judge, comparing self-training gains;
- **E2**: frozen judge vs judge updated with training, comparing degeneration rates;
- **E3**: pre-registered stopping vs model-decided stopping, measuring early-stop rates;
- **E4**: self-collected-only data vs mixed human data, comparing distribution narrowing;
- **E5**: whether capability keeps rising over multiple rounds or plateaus.

## 8. Related Work

- RSI and seed AI: Good (1965), Yudkowsky (2008);
- Gödel machine: Schmidhuber (2003), which requires formal proofs — far stronger and far harder than pseudo-RSI;
- Self-training and rejection sampling: STaR (Zelikman et al., 2022), rejection-sampling fine-tuning;
- Self-improvement and self-rewarding: Self-Rewarding LMs (Yuan et al., 2024), SPIN (Chen et al., 2024);
- Self-refinement: Self-Refine (Madaan et al., 2023);
- LLM-as-a-judge: MT-Bench / LLM-as-a-Judge (Zheng et al., 2023);
- Human preferences and reward models: Christiano et al. (2017), InstructGPT (Ouyang et al., 2022);
- Generative adversarial training: the GAN discriminator/generator (Goodfellow et al., 2014);
- Model collapse: Shumailov et al. (2024);
- Reward hacking and Goodhart: Amodei et al. (2016), Skalse et al. (2022);
- Weak-to-strong generalization: Burns et al. (2023).

## 9. Roadmap

| Version | Goal | Criterion |
| --- | --- | --- |
| v0.1 | judge calibration | agreement with human preference well above chance |
| v0.2 | single-round self-training | stable gains on held-out benchmarks |
| v0.3 | multi-round + audit | judge does not drift, capability keeps rising |
| v0.4 | stopping rule | pre-registered stop agrees with human judgment |
| v1.0 | end-to-end report | labor savings and capability gains quantified |

## 10. TODO

- [ ] Define machine-checkable standard templates
- [ ] Build the human anchor set
- [ ] Implement judge freezing/reset mechanisms
- [ ] Design pre-registered stopping rules
- [ ] Measure distribution diversity after multiple rounds

## 11. Glossary

| Term | Meaning |
| --- | --- |
| Pseudo-RSI | an improvement loop driven by external standards, with model self-judgment and self-training, and humans still in the loop |
| Judge | the model or program that evaluates artifacts against the standard |
| Discrimination | the judge's probability of ranking good over bad, minus the random baseline |
| Trajectory contamination | optimization history changing the judge's scores |
| Anchor set | fixed human-labeled samples used for calibration and acceptance |

## 12. References

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

## 13. Changelog

| Date | Change |
| --- | --- |
| 2026-11-28 | First draft: pseudo-RSI positioning, judgment/execution separation precondition, standard-driven loop, trajectory contamination and stopping bias, defects and evaluation |
