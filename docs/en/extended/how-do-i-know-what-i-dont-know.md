# How Do I Know What I Don't Know (Extended)

> Stage: **In Progress** (~7 pages) · Status: **WIP** · Last updated: 2026-11-26
> Language: [中文](../../扩展中途版/我如何知道我不知道什么.md) · English

## 1. The Problem

### 1.1 The Machine's "Ignorance Blind Spot"

Models can answer many questions, but rarely say "I don't know this" in a reliable way. Worse, they often give confident answers when they do not know. We call this missing capability the **ignorance blind spot**:

- not knowing a fact, yet producing a plausible-looking answer;
- not knowing that a domain exists (unknown unknowns), and therefore never asking;
- failing to judge whether a problem is beyond one's capability (long-horizon planning, real-time information, tool use);
- being unable to explain "why I am uncertain" when pressed.

For humans, knowing what one does not know is part of common sense: most people can roughly judge whether they understand a topic; a minority systematically overestimate themselves (Kruger & Dunning, 1999). **Humans generally have this ability; machines have partial research on calibration, abstention, and selective prediction (Kadavath et al., 2022; Yin et al., 2023; Cheng et al., 2024), but it has not yet become the standard design of general systems, and active discovery of unknown unknowns is especially missing.**

### 1.2 The Human Reference

- **Metamemory**: the tip-of-the-tongue feeling — "I can't recall it, but I know it exists" (Nelson & Narens, 1990);
- **Calibration**: confidence in an answer roughly matches accuracy;
- **External help**: when unsure, look it up or ask someone — the feeling of ignorance triggers action;
- **Boundary awareness**: knowing one's own specialty, and which questions should be referred elsewhere.

What machines need is not "speaking more cautiously" but **explicit ignorance representation + executable actions**.

### 1.3 Why Current Models Lack This Capability

1. The training objective is next-token probability, rewarding "continuations that look like correct answers", not "admitting ignorance";
2. parametric knowledge is implicit and entangled; there is no queryable "knowledge boundary" inside the model;
3. alignment training favors helpful answers; refusal is often treated as failure (Ouyang et al., 2022);
4. evaluation looks at accuracy, not "do you know that you are guessing";
5. unknown unknowns have no samples in the training distribution — the model cannot learn from "failures it has never seen".

## 2. Defining Three Kinds of Ignorance

| Type | Definition | Internally detectable? |
| --- | --- | --- |
| Known knowns | Know it, and can answer correctly | n/a |
| Known unknowns | Know that I don't know: missing facts, missing evidence, insufficient capability | Mostly detectable: calibration, retrieval, consistency |
| Unknown unknowns | Don't know that I don't know: out-of-domain, blind spots, novel problems | Not by introspection alone; requires external probing |

Also distinguish three boundaries:

- **Knowledge boundary**: whether a fact or domain exists in the training data and parameters;
- **Capability boundary**: even with the knowledge, can it be done (reasoning length, tools, real-time access, reliability);
- **Coverage boundary**: the gap between the scope the system claims and the scope it actually covers.

## 3. Core Claims

**Claim 1: ignorance must be a first-class citizen.** The output should not be just an answer $a$, but a pair $(a, r)$, where $r$ is an ignorance report: confidence, gap type, evidence, and recommended action.

**Claim 2: detectability must be layered.** Known unknowns can be detected with internal signals plus external evidence; unknown unknowns cannot be self-certified and can only be *discovered* through **coverage assumptions + external probing + feedback loops**. The design must honestly separate the two layers instead of pretending introspection solves everything.

**Claim 3: ignorance must drive action.** Once ignorance is detected, the next step is to ask, retrieve, abstain, escalate, or log for later investigation — not to keep generating.

**Claim 4: ignorance awareness ultimately leads to spontaneous exploration.** Passive detection works only when asked; the system must also be able to find gaps on its own, judge whether they are worth filling, and execute exploration actions (asking, retrieval, experiments). Otherwise "knowing that I don't know" never becomes "no longer not knowing".

## 4. Design: An Ignorance-Aware System

### 4.1 The Ignorance Report

Every output carries a structured report instead of a bare answer:

```json
{
  "answer": "...",
  "confidence": 0.62,
  "gap_type": "known_unknown | suspected_unknown | capability_limit | out_of_scope",
  "evidence": ["no retrieval results", "two samples disagree", "beyond the declared knowledge cutoff"],
  "scope": {"domain": "...", "as_of": "2026-06"},
  "action": "answer | ask_clarification | retrieve | abstain | escalate",
  "probe": "suggested external verification"
}
```

The report itself needs a confidence: the system's judgment that it "doesn't know" can also be wrong.

### 4.2 Detecting Known Unknowns

| Signal | Purpose | Limitation |
| --- | --- | --- |
| Temperature scaling / calibration (Guo et al., 2017) | Make confidence ≈ accuracy | fails out of distribution |
| Selective prediction (Geifman & El-Yaniv, 2017) | Risk–coverage curve, abstention threshold | needs held-out data |
| Consistency probes | Multiple samples or paraphrases; high disagreement means uncertainty | costly |
| Evidence checks | Empty retrieval, missing citations, tool failure | depends on external system quality |
| Conformal prediction (Vovk et al., 2005; Angelopoulos & Bates, 2021) | Prediction sets with coverage guarantees under exchangeability | guarantee breaks when the assumption fails |
| Error-prediction head (Kadavath et al., 2022; Yin et al., 2023) | Hidden states, entropy, retrieval scores predict "will this answer be wrong" | sensitive to distribution shift |

### 4.3 Discovering Unknown Unknowns

This is the hardest part. The principle: **do not claim to introspect unknown unknowns; design discovery mechanisms instead.**

- **Explicit coverage assumptions**: the system declares the time, domains, languages, and tools of its training data; anything outside is marked `out_of_scope` instead of guessed;
- **Active self-testing**: generate probe questions it "should / should not" be able to answer, verify with external knowledge bases or tools; failures are discovered blind spots;
- **Distribution probes**: OOD detection (Hendrycks & Gimpel, 2017), topic clustering, new-word discovery to flag "things never seen";
- **Failure logs and feedback loops**: user corrections, tool errors, retrieval contradictions are recorded in the self-knowledge base; online failures are the most authentic source of unknown unknowns;
- **Red teaming and audits**: periodically audit with adversarial and cross-domain questions and measure the discovery rate.

### 4.4 The Self-Knowledge Base

Maintain an updatable "capability card":

- what I know: domains, languages, tools, freshness;
- what I don't know: known failure modes, refusal list;
- where I am likely to err: sub-distributions with high error rates;
- the time and result of the latest update and audit.

It connects naturally to online learning (see *Fast Prediction Framework*, 9.3): new data and new failures feed this card, and the card grows with the system.

### 4.5 The Action Layer

| Gap type | Action |
| --- | --- |
| Known unknown (lookup possible) | retrieve / use tools / ask a clarifying question |
| Known unknown (not lookup-able) | say clearly that I don't know, with confidence and reasons |
| Suspected unknown unknown | declare scope, request external verification, mark for audit |
| Capability boundary | degrade gracefully: decompose the task, use tools, escalate to a human |

## 5. An Associated Question: Spontaneous Exploration of Unknown Domains

So far we have discussed "ignorance judgment when asked". There is a stronger question: **can a machine discover what it lacks, and go and fill the gap on its own?**

### 5.1 Why Passive Detection Is Not Enough

- Unknown unknowns do not appear in queries by themselves; the system usually discovers its blind spots only *after* a failure;
- evaluation sets and red-team audits are externally driven, and their coverage is limited by what humans think to ask;
- in deployment, new domains, new terminology, and new events keep appearing, so the system must notice "I may not understand this" by itself.

A system that only answers but never explores has passive ignorance awareness: it is exposed only when a user runs into the gap. Truly knowing what one does not know requires **active discovery + active judgment**.

### 5.2 Components

1. **Gap sensing**: find candidate blind spots from failure logs, coverage maps, retrieval contradictions, and new-word/new-topic signals;
2. **Value judgment**: estimate each candidate's information value (expected information gain, task relevance, risk) and decide whether it is worth exploring;
3. **Exploration actions**: ask the user, retrieve, call tools, run experiments, generate self-tests;
4. **Outcome judgment**: after exploring, verify whether the gap is actually filled (external evidence, consistency re-checks) and update the self-knowledge base;
5. **Exploration discipline**: budget limits and safety boundaries; avoid aimless wandering and self-confirmation.

### 5.3 Formalization

View exploration as choosing an action $a$ that maximizes the reduction in ignorance minus cost and risk:

$$
a^* = \arg\max_{a \in \mathcal{A}} \; \mathbb{E}\big[\Delta I(a)\big] - \lambda \cdot \mathrm{cost}(a) - \mu \cdot \mathrm{risk}(a)
$$

where $\Delta I(a)$ is the reduction in uncertainty about the blind spots after the action, and $\mathcal{A}$ is the set of available information-acquisition actions. Bayesian active-learning information gain (Houlsby et al., 2011) can serve as a proxy.

Difficulty: $\Delta I(a)$ cannot be estimated accurately — otherwise the answer would already be known. Only proxies are available: retrieval hit rate, question answerability, user feedback, degree of contradiction with existing knowledge.

### 5.4 Relation to the Ignorance-Aware System

- the `probe` field of the ignorance report is the entry point for exploration;
- the self-knowledge base (4.4) is the exploration map: it records covered, uncovered, and suspected blind spots;
- the audit loop (4.3) supplies external discovery sources; spontaneous exploration turns external drive into internal drive;
- online learning (Fast Prediction Framework, 9.3) provides the update channel: newly acquired knowledge must be able to land in the base.

### 5.5 Risks

- **Runaway exploration**: endless questioning or retrieval consumes budget and user patience;
- **Self-confirmation**: verifying blind spots with one's own knowledge drifts further off;
- **Safety boundaries**: autonomous exploration may trigger external actions (tool calls, requests) and needs permissions and auditing;
- **Evaluation lag**: the value of exploration often shows up late, and short-term metrics underestimate it.

### 5.6 Evaluation

- discovery rate: new blind spots found per unit budget;
- relevance: how related the discovered blind spots are to the task/user needs;
- convergence: does coverage rise and repeated exploration fall after multiple rounds;
- safety: rate of unauthorized actions and harmful exploration.

## 6. Training

- **Error-prediction head**: label answers as correct/incorrect on held-out data, classify with hidden states, entropy, and retrieval features; watch for distribution shift;
- **Conformal calibration**: fit nonconformity scores on validation data; emit prediction sets with coverage guarantees at inference;
- **Unknown-sample construction**: sample from outside the training distribution (new topics, new languages, new time periods), label them "unknown", and train refusals and scope declarations;
- **Red-team data**: adversarial questions, hallucination bait, long-tail facts;
- **Online loop**: online failure → log → retraining → self-knowledge-base update;
- **Curriculum**: first learn "say yes when I can, no when I can't", then learn "discover what I don't know".

Risk: self-supervised signals alone reinforce self-confirmation bias; external labels are mandatory — tool verification, human feedback, retrieval results.

## 7. Evaluation

| Dimension | Metric | Description |
| --- | --- | --- |
| Calibration | ECE, Brier, reliability diagram | does confidence match accuracy |
| Selective prediction | risk–coverage curve, AUROC | can abstention filter out errors |
| Known-unknown recall | detection rate of questions that should be refused | misses vs false alarms |
| Unknown-unknown discovery | new blind spots found per audit round, discovery latency | driven by external audits |
| Scope-claim accuracy | gap between declared and real capability | prevents "fake coverage" |
| Action quality | gain from asking / retrieval / escalation | does ignorance turn into a better outcome |
| User trust | appropriate reliance | do users rely correctly when the system is uncertain |

Key experiments:

- risk–coverage curves on the known-unknown set, against "always answer" and "always refuse" baselines;
- an unknown-unknown test set (domains never seen in training) to check whether the system marks them `out_of_scope`;
- multi-round red-team audits to see whether blind spots decrease;
- ablations: degradation when retrieval, consistency, or conformal signals are removed.

## 8. Relation to the Fast Prediction Framework

- Module 1's coverage (Recall@k, FullCoverageRate) is essentially a **measurable known unknown**: insufficient coverage means the system knows it may not be able to answer fully;
- the coverage constraint in triple consistency can be exposed directly in the ignorance report;
- genuine unknown unknowns still require external probing and the online-learning loop; compressed-vocabulary statistics alone cannot find them.

## 9. Difficulties and Open Questions

- **The meta-ignorance paradox**: judging "I don't know" can itself be wrong; how do we calibrate judgments about ignorance?
- **Unknown unknowns cannot be self-certified**: every internal detector has blind spots, so external audits are required — and audits can miss things too;
- **Over-refusal**: saying "I don't know" too often destroys utility; the utility–risk trade-off must be explicit;
- **Credibility of scope claims**: a declared "I know domain A" may not match reality;
- **Evaluation lag**: online failures are often noticed long after they happen;
- **Safety**: ignorance reports may be misread as reliability guarantees.

## 10. Related Work

- Calibration and selective prediction: temperature scaling, risk–coverage, selective classification (Guo et al., 2017; Geifman & El-Yaniv, 2017);
- conformal prediction: distribution-free coverage guarantees (Vovk et al., 2005; Angelopoulos & Bates, 2021);
- OOD detection and open-world learning (Hendrycks & Gimpel, 2017);
- hallucination detection and factuality evaluation (Ji et al., 2023);
- metacognition and metamemory (Nelson & Narens, 1990; Kruger & Dunning, 1999);
- active learning, information acquisition, and clarification questions (Settles, 2009; Houlsby et al., 2011);
- epistemic uncertainty (Hüllermeier & Waegeman, 2021).

## 11. Roadmap

| Version | Goal | Criterion |
| --- | --- | --- |
| v0.1 | Calibration + abstention baseline | ECE and risk–coverage better than naive baselines |
| v0.2 | Ignorance-report schema + retrieval/consistency signals | known-unknown recall ≥ 90% |
| v0.3 | Self-knowledge base and audit loop | each audit round finds new failure modes |
| v0.4 | Online loop | online-failure-to-update latency acceptable |
| v0.5 | Spontaneous exploration prototype | discovers and verifies new blind spots per unit budget |
| v1.0 | End-to-end system | significantly better than "always answer" on the unknown-unknown test set |

## 12. TODO

- [ ] Work out calibration and conformal implementations
- [ ] Design the unknown-unknown test set
- [ ] Define the ignorance-report interface
- [ ] Small prototype: error-prediction head + retrieval signals
- [ ] Design triggers and budget policy for spontaneous exploration
- [ ] Connect with the Fast Prediction Framework's coverage metrics

## 13. References

1. Kruger, J., & Dunning, D. (1999). Unskilled and Unaware of It: How Difficulties in Recognizing One's Own Incompetence Lead to Inflated Self-Assessments. *Journal of Personality and Social Psychology*, 77(6).
2. Nelson, T. O., & Narens, L. (1990). Metamemory: A Theoretical Framework and New Findings. *Psychology of Learning and Motivation*, 26.
3. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.
4. Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. *NeurIPS 2017*.
5. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
6. Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.
7. Kadavath, S., Conerly, T., Askell, A., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
8. Yin, Z., Wang, Q., & Zhang, X. (2023). Do Large Language Models Know What They Don't Know? *ACL 2023 Findings*.
9. Cheng, Q., Sun, T., Zhang, W., et al. (2024). Can AI Assistants Know What They Don't Know? *ICML 2024*.
10. Hendrycks, D., & Gimpel, K. (2017). A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. *ICLR 2017*.
11. Ji, Z., Lee, N., Frieske, R., et al. (2023). Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*, 55(12).
12. Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS 2022*.
13. Settles, B. (2009). Active Learning Literature Survey. *Computer Sciences Technical Report 1648*, University of Wisconsin–Madison.
14. Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). Bayesian Active Learning for Classification and Preference Learning. arXiv:1112.5745.
15. Hüllermeier, E., & Waegeman, W. (2021). Aleatoric and Epistemic Uncertainty in Machine Learning: An Introduction to Concepts and Methods. *Machine Learning*, 110.

## 14. Changelog

| Date | Change |
| --- | --- |
| 2026-11-26 | First draft: problem, three kinds of ignorance, ignorance-report design, training and evaluation |
| 2026-11-28 | Review: restored the accidentally deleted "Claim 3"; softened the overclaim that machines lack the capability entirely; added citations on calibration, selective prediction, conformal prediction, and self-knowledge |
| 2026-11-26 | Added §5: spontaneous exploration of unknown domains — components, formalization, risks, evaluation; added Claim 4 and a roadmap row |
