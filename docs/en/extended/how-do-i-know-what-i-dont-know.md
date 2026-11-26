# How Do I Know What I Don't Know (Extended)

> Stage: **In Progress** (~5 pages) · Status: **WIP** · Last updated: 2026-11-26
> Language: [中文](../../扩展中途版/我如何知道我不知道什么.md) · English

## 1. The Problem

### 1.1 The Machine's "Ignorance Blind Spot"

Models can answer many questions, but rarely say "I don't know this" in a reliable way. Worse, they often give confident answers when they do not know. We call this missing capability the **ignorance blind spot**:

- not knowing a fact, yet producing a plausible-looking answer;
- not knowing that a domain exists (unknown unknowns), and therefore never asking;
- failing to judge whether a problem is beyond one's capability (long-horizon planning, real-time information, tool use);
- being unable to explain "why I am uncertain" when pressed.

For humans, knowing what one does not know is part of common sense: most people can roughly judge whether they understand a topic; a minority systematically overestimate themselves (the Dunning–Kruger effect). **Humans at least generally have this ability; machines have not treated it as a capability that must be designed.**

### 1.2 The Human Reference

- **Metamemory**: the tip-of-the-tongue feeling — "I can't recall it, but I know it exists";
- **Calibration**: confidence in an answer roughly matches accuracy;
- **External help**: when unsure, look it up or ask someone — the feeling of ignorance triggers action;
- **Boundary awareness**: knowing one's own specialty, and which questions should be referred elsewhere.

What machines need is not "speaking more cautiously" but **explicit ignorance representation + executable actions**.

### 1.3 Why Current Models Lack This Capability

1. The training objective is next-token probability, rewarding "continuations that look like correct answers", not "admitting ignorance";
2. parametric knowledge is implicit and entangled; there is no queryable "knowledge boundary" inside the model;
3. alignment training favors helpful answers; refusal is often treated as failure;
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
| Temperature scaling / calibration | Make confidence ≈ accuracy | fails out of distribution |
| Selective prediction | Risk–coverage curve, abstention threshold | needs held-out data |
| Consistency probes | Multiple samples or paraphrases; high disagreement means uncertainty | costly |
| Evidence checks | Empty retrieval, missing citations, tool failure | depends on external system quality |
| Conformal prediction | Prediction sets with coverage guarantees under exchangeability | guarantee breaks when the assumption fails |
| Error-prediction head | Hidden states, entropy, retrieval scores predict "will this answer be wrong" | sensitive to distribution shift |

### 4.3 Discovering Unknown Unknowns

This is the hardest part. The principle: **do not claim to introspect unknown unknowns; design discovery mechanisms instead.**

- **Explicit coverage assumptions**: the system declares the time, domains, languages, and tools of its training data; anything outside is marked `out_of_scope` instead of guessed;
- **Active self-testing**: generate probe questions it "should / should not" be able to answer, verify with external knowledge bases or tools; failures are discovered blind spots;
- **Distribution probes**: OOD detection, topic clustering, new-word discovery to flag "things never seen";
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

## 5. Training

- **Error-prediction head**: label answers as correct/incorrect on held-out data, classify with hidden states, entropy, and retrieval features; watch for distribution shift;
- **Conformal calibration**: fit nonconformity scores on validation data; emit prediction sets with coverage guarantees at inference;
- **Unknown-sample construction**: sample from outside the training distribution (new topics, new languages, new time periods), label them "unknown", and train refusals and scope declarations;
- **Red-team data**: adversarial questions, hallucination bait, long-tail facts;
- **Online loop**: online failure → log → retraining → self-knowledge-base update;
- **Curriculum**: first learn "say yes when I can, no when I can't", then learn "discover what I don't know".

Risk: self-supervised signals alone reinforce self-confirmation bias; external labels are mandatory — tool verification, human feedback, retrieval results.

## 6. Evaluation

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

## 7. Relation to the Fast Prediction Framework

- Module 1's coverage (Recall@k, FullCoverageRate) is essentially a **measurable known unknown**: insufficient coverage means the system knows it may not be able to answer fully;
- the coverage constraint in triple consistency can be exposed directly in the ignorance report;
- genuine unknown unknowns still require external probing and the online-learning loop; compressed-vocabulary statistics alone cannot find them.

## 8. Difficulties and Open Questions

- **The meta-ignorance paradox**: judging "I don't know" can itself be wrong; how do we calibrate judgments about ignorance?
- **Unknown unknowns cannot be self-certified**: every internal detector has blind spots, so external audits are required — and audits can miss things too;
- **Over-refusal**: saying "I don't know" too often destroys utility; the utility–risk trade-off must be explicit;
- **Credibility of scope claims**: a declared "I know domain A" may not match reality;
- **Evaluation lag**: online failures are often noticed long after they happen;
- **Safety**: ignorance reports may be misread as reliability guarantees.

## 9. Related Work

- Calibration and selective prediction: temperature scaling, risk–coverage, selective classification;
- conformal prediction: distribution-free coverage guarantees;
- OOD detection and open-world learning;
- hallucination detection and factuality evaluation;
- metacognition and metamemory (cognitive science);
- active learning, information acquisition, and clarification questions.

## 10. Roadmap

| Version | Goal | Criterion |
| --- | --- | --- |
| v0.1 | Calibration + abstention baseline | ECE and risk–coverage better than naive baselines |
| v0.2 | Ignorance-report schema + retrieval/consistency signals | known-unknown recall ≥ 90% |
| v0.3 | Self-knowledge base and audit loop | each audit round finds new failure modes |
| v0.4 | Online loop | online-failure-to-update latency acceptable |
| v1.0 | End-to-end system | significantly better than "always answer" on the unknown-unknown test set |

## 11. TODO

- [ ] Work out calibration and conformal implementations
- [ ] Design the unknown-unknown test set
- [ ] Define the ignorance-report interface
- [ ] Small prototype: error-prediction head + retrieval signals
- [ ] Connect with the Fast Prediction Framework's coverage metrics

## 12. Changelog

| Date | Change |
| --- | --- |
| 2026-11-26 | First draft: problem, three kinds of ignorance, ignorance-report design, training and evaluation |
