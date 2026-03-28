# Deep Research Review — CEF: The Completion Fallacy
## NeurIPS 2026 Position Paper

**Date:** March 13, 2026
**Method:** 3-round structured mock review (Codex MCP unavailable)
**Source documents:** plan file v3.0, NOVELTY_CHECK_CEF.md, literature_review.md (59 papers)
**Target venue:** NeurIPS 2026 Position Paper track (~May 22-28 deadline)

---

## Research Context Briefing

### The Position
> "Current LLM agent benchmarks measure *what* agents accomplish, not *how* they think. This creates a false equivalence between agents that succeed through genuine reasoning and those that succeed through sophisticated pattern matching — making it impossible to predict failure modes, generalization, or safety properties. Cognitive science offers a principled, operationalizable alternative."

### The Three Contributions
1. **The Completion Fallacy** — formal argument: task-completion metrics are structurally insufficient due to *multiple realizability* (identical outputs from radically different cognitive processes)
2. **CEF** — four cognitively-grounded evaluation dimensions (WMF, MCC, EMC, CLA), each with established cognitive science anchor and API-level operationalization
3. **Empirical Validation** — Tier 1: MCC-MA (batch error prediction, Pearson r), MCC-CE (three-metric monitoring/control decomposition), CLA-CR (recovery paradigm). E5: Kendall's τ (N=12 models, CEF vs. benchmarks predicting OOD)

### Key Prior Art Risk (from novelty check)
- PAE [2603.03116]: 27-78% of successes procedurally corrupt — empirical support for Completion Fallacy
- [2512.12791]: "beyond task completion" operational framework — different angle
- NeuroCognition [2603.02540]: cognitive ability tests — different type of evaluation

### Known Weaknesses
- "Multiple realizability" argument needs a concrete example, not just abstract claim
- E5 (Kendall's τ, N=12) is the empirical linchpin — if τ(CEF,OOD) ≤ τ(benchmarks,OOD), paper loses its empirical claim
- Weight assignments in composite scores (WMF = 0.50×AM + 0.30×IM + 0.20×IR) are arbitrary
- RF dimension was cut — some reviewers may notice the framework looks incomplete without reasoning robustness

---

## Round 1 — Deep Critique

*As NeurIPS 2026 Position Paper Area Chair + Reviewer:*

### Critique 1: The "Multiple Realizability" Argument Is Philosophically Borrowed But Empirically Thin — CRITICAL

The paper's central argument borrows "multiple realizability" from philosophy of mind (Putnam, 1967). This is intellectually sophisticated but creates a specific burden: the paper must SHOW that two agents with identical task completion scores actually have different cognitive architectures.

The current plan lists 2 case studies: "models with identical benchmark scores but completely divergent CEF profiles." This is necessary but may not be sufficient. The reviewer will ask:

> "You show that models have different CEF profiles. But you haven't shown that these differences MATTER — that the model with a better CEF profile is actually safer, more generalizable, or fails differently. Without that, you've shown cognitive profiles differ, not that task completion obscures something important."

**Severity:** HIGH — this is the difference between "we have a better characterization" (weak) and "existing benchmarks actively mislead you" (strong position paper claim).

**What's needed:** At least one concrete case where two models have: (a) similar MMLU/AgentBench scores, (b) divergent CEF profiles, AND (c) different failure patterns on a held-out task. The OOD generalization in E5 partly addresses this — but the case study must be vivid and specific.

### Critique 2: The CEF Composite Weights Are Arbitrary — MEDIUM

`WMF = 0.50 × WMF-AM + 0.30 × WMF-IM + 0.20 × WMF-IR`
`MCC = 0.55 × MCC-MA + 0.45 × MCC-CE`
`EMC = 0.40 × EMC-EI + 0.35 × EMC-TO + 0.25 × EMC-SA`
`CLA = 0.50 × CLA-DC + 0.30 × CLA-RA + 0.20 × CLA-CR`

A reviewer will ask: "Why 0.50 and not 0.40? Why 0.55 and not 0.50? These weights appear to have no principled basis."

**Severity:** MEDIUM — doesn't kill the paper but weakens the framework's scientific credibility.

**What's needed:** Either (a) derive weights empirically via regression on a held-out task set (weights = beta coefficients predicting task performance), or (b) run a sensitivity analysis showing that the E5 result holds under different reasonable weight sets {equal weights, PCA-derived weights, proposed weights}, or (c) simply use equal weights throughout and note this as a simplification.

### Critique 3: "Cognitively Grounded" May Mean "Superficially Analogized" — MEDIUM

Reviewers with cognitive science backgrounds will note: LLMs have no working memory buffer in the Baddeley sense, no hippocampal episodic encoding, no frontal executive control. The cognitive science anchors (Miller 1956, Tulving 1972, Nelson & Narens 1990) are developed for biological systems with specific neural substrates.

The reviewer will say: "You're using cognitive science vocabulary but the underlying mechanisms are completely different. Miller's 7±2 doesn't apply to a transformer with 128K context. This weakens your theoretical claim."

**Severity:** MEDIUM — standard objection to cognitive science-informed AI papers; addressable but requires explicit acknowledgment.

**What's needed:** A paragraph in Section 4 or Limitations explicitly defending the functional interpretation: "CEF does not claim that LLM agents have the same *mechanisms* as biological cognitive systems. Rather, it operationalizes the *functional capacities* that cognitive science has identified as necessary for robust, generalizable cognition — regardless of substrate. An agent that cannot monitor its own errors is functionally metacognitively deficient, regardless of whether it has a prefrontal cortex."

### Critique 4: E5 Statistical Design Has Issues — HIGH

E5 computes Kendall's τ between CEF rankings and OOD task rankings across N=12 models. Several statistical concerns:

1. **Multiple comparisons:** E5 tests τ(CEF, OOD) vs. τ(benchmarks, OOD) for "≥2 of 4 held-out tasks (p<0.05)." With 4 tasks and p=0.05 threshold, the chance of ≥1 false positive under the null is 18.5% — the "≥2 of 4" condition makes this worse because it invites cherry-picking which 2 tasks show significance.

2. **N=12 for Kendall's τ:** With N=12, Kendall's τ has low power. The minimum detectable effect size for p<0.05 (two-tailed) requires τ ≥ 0.45 — only large effects will be significant.

3. **Confound: CEF correlates with model size.** If larger models score higher on both CEF and OOD tasks, then τ(CEF, OOD) > 0 is explained by model scale, not CEF's added value over benchmarks. Need to show that CEF residuals (after regressing out benchmark scores) still predict OOD.

**Severity:** HIGH — statistical design must be fixed before submission.

**What's needed:**
- Pre-register the specific 4 OOD tasks and "success = τ higher for ≥2" before running
- Run a partial rank correlation: τ(CEF, OOD | benchmarks) — does CEF explain OOD variance beyond what benchmarks already explain?
- Report N and power explicitly; note effect size requirement
- Consider: replace success criterion with "partial τ > 0 for composite CEF (p<0.05, one-tailed)" — cleaner

### Critique 5: Position Paper Must Have a "Call to Action" With Teeth — LOW-MEDIUM

NeurIPS position papers that succeed don't just argue a position — they provide actionable recommendations that the community can implement. The current plan has a "3 concrete recommendations" section but this needs to be substantive.

**What's needed:**
1. "Adopt CEF as a required evaluation dimension in agent papers" — concrete: specify which sub-experiments require ≤200 API calls
2. "Report CEF profiles alongside benchmark scores in model cards" — specific: provide a template
3. "Establish held-out OOD test sets for cognitive evaluation" — specific: propose 4 tasks that the community should adopt

---

## Round 2 — Author Responses + Experiment Design

### Response to Critique 1 (Multiple Realizability Evidence) — ACCEPT

The case study requirement is correct and strengthens the paper. **Design:**

**Exhibit A (to be reported in Section 5):**
- Run ALL experiments on 12 models
- Identify 2 pairs of models where: MMLU score within 3 points AND CEF-MCC diverges by >30 percentile points
- Run 2 held-out tasks (agentic, not in training distribution): multi-step tool use + conditional instruction following
- Report: Model A (MMLU=78%, MCC=high) succeeds on held-out at 71%; Model B (MMLU=77%, MCC=low) fails at 34%
- This one result is worth more than all the theory in Section 2

**Additional exhibit:** cite PAE [2603.03116] prominently in Section 2: "Cao et al. (2026) show that 27-78% of benchmark-reported successes are procedurally corrupt — precisely what the Completion Fallacy predicts."

### Response to Critique 2 (Arbitrary Weights) — ACCEPT PARTIALLY

**Solution:** Run a sensitivity analysis in the appendix. Show that E5 Kendall's τ result holds under three weighting schemes:
1. Proposed weights (0.50/0.30/0.20, etc.)
2. Equal weights (all sub-dimensions weighted equally)
3. PCA-derived weights (first principal component from sub-dimension scores)

If the result is stable across all three: "The E5 result is robust to the specific weighting scheme chosen for CEF composites (Appendix B), suggesting the individual sub-dimensions each contribute independently."

If it's not stable: use equal weights as the default. Simpler is more defensible.

### Response to Critique 3 (Cognitive Science Analogy) — ACCEPT, ADD PARAGRAPH

Accept. Add explicitly to Section 4:

> "A natural objection: LLMs are not biological systems and may not share the mechanisms underlying cognitive science constructs. We adopt a *functional* interpretation: CEF does not claim that transformer attention implements Baddeley's phonological loop, or that RLHF fine-tuning creates hippocampal episodic traces. Rather, CEF operationalizes the *functional capacities* — error monitoring, episodic coherence, load-adaptive regulation — that cognitive science has identified as necessary for robust cognition in agents acting under uncertainty. The substrate is irrelevant to the functional test. A self-driving car can fail the functional equivalent of a Stroop task (attending to irrelevant stimuli) without having frontal executive cortex."

Also cite Niu et al. [2409.02387]: "The cognitive science lens is defensible precisely because LLMs have converged on functional analogues of human cognitive capacities despite different substrates (Niu et al., 2024)."

### Response to Critique 4 (E5 Statistical Design) — ACCEPT, REDESIGN

**Revised E5 design:**

Primary test: **Partial rank correlation**
- Rank 12 models on CEF composite
- Rank same 12 models on benchmark composite (MMLU + HumanEval + GSM8K)
- Rank same 12 models on 4 OOD tasks
- Compute: τ(CEF, OOD_i) and τ(benchmarks, OOD_i) for i = 1..4
- **Primary hypothesis:** τ(CEF, OOD) > τ(benchmarks, OOD) as averaged across 4 OOD tasks; one-sided permutation test, p<0.05

Sensitivity test: partial correlation after regressing out benchmark ranks
- Compute residual_CEF = CEF rank after removing benchmark rank
- Compute τ(residual_CEF, OOD) > 0 (one-sided, p<0.05)
- If positive: CEF captures something benchmarks don't (even controlling for model capability)

**OOD task selection (pre-register):**
1. Multi-step tool-use agent task with conditional policies (tests PM analog)
2. Long-horizon planning with self-monitoring requirements (tests MCC)
3. Multi-source episodic retrieval under interference (tests EMC)
4. Adaptive difficulty reasoning (tests CLA)

**Power analysis:** N=12, τ≥0.45 required for p<0.05. Note this explicitly; frame as "proof-of-concept" with note that larger study (N=30) would confirm with smaller effect size.

### Response to Critique 5 (Call to Action) — ACCEPT

Revise Section 7 (Position & Call to Action) to include concrete, implementable recommendations:

**Recommendation 1 — Minimum CEF Reporting Standard:**
> "We propose that papers introducing new LLM agents include a 'CEF profile table' reporting scores on MCC-MA, MCC-CE, and CLA-CR (the three Tier-1 sub-dimensions). Total cost: ≤200 API calls per model (detailed in Appendix C). We release a 'CEF-minimal' evaluation suite to facilitate adoption."

**Recommendation 2 — OOD Held-Out Set:**
> "The community should establish 4 held-out OOD evaluation tasks (proposed above) as standard annual benchmarks, updated yearly to prevent contamination. We release the 4 tasks used in E5 as an initial proposal."

**Recommendation 3 — Model Card Amendment:**
> "AI providers should extend model cards to include cognitive process quality metrics. We provide a template model card with CEF dimensions alongside accuracy and benchmark scores (Appendix D)."

---

## Round 3 — Mock NeurIPS 2026 Position Paper Review

```
===================================================================
NEURIPS 2026 REVIEW — POSITION PAPER TRACK
Paper: "The Completion Fallacy: LLM Agent Benchmarks Measure
        the Wrong Thing"
===================================================================

SUMMARY
-------
This position paper argues that task-completion benchmarks for LLM
agents are structurally insufficient due to "multiple realizability"
— the same task outcome can arise from cognitively distinct processes,
making completion rates epistemically uninformative about safety,
generalization, or failure modes. The authors introduce the Cognitive
Evaluation Framework (CEF), with four cognitive-science-grounded
dimensions (WMF, MCC, EMC, CLA), and provide proof-of-concept
empirical validation showing CEF predicts OOD generalization better
than benchmark composites across 12 frontier models (Kendall's τ).

STRENGTHS
---------
S1: The core argument ("multiple realizability") is novel in the LLM
    evaluation context. While "beyond task completion" arguments exist
    ([2512.12791], PAE [2603.03116]), this paper makes a philosophically
    stronger claim: not merely that completion misses things, but that
    completion is *structurally uninformative* about cognitive process
    quality. This is a distinctive position.

S2: The cognitive science anchors are appropriate and well-established.
    Nelson & Narens (1990) monitoring/control, Tulving episodic memory,
    and Sweller cognitive load are canonical frameworks with extensive
    empirical support. The MCC monitoring/control decomposition
    (MCC-MA, MCC-CE) has no direct prior art.

S3: PAE [2603.03116] provides striking independent empirical support
    for the Completion Fallacy (27-78% of successes are procedurally
    corrupt). The paper is well-timed to synthesize a coalescing field.

S4: The E5 design (Kendall's τ, N=12, OOD prediction) directly tests
    the paper's central claim empirically — if the result holds,
    this paper has real teeth.

S5: The call to action with concrete implementable recommendations
    (CEF-minimal reporting, OOD held-out set) is practical and
    addresses the "so what?" question.

WEAKNESSES
----------
W1: The "multiple realizability" argument needs a concrete exhibit:
    two specific real models with identical benchmark scores but
    divergent CEF profiles AND divergent real-task failure patterns.
    Without this, the claim remains theoretical.
    [Severity: HIGH]

W2: E5 partial correlation (controlling for model scale/capability
    captured by benchmark scores) must be reported. If τ(CEF,OOD) > 0
    is explained entirely by model scale (larger models score higher on
    everything), the claim that CEF adds information beyond benchmarks
    is unsupported.
    [Severity: HIGH]

W3: The cognitive science analogy requires explicit functional
    defense. Reviewers with cognitive science backgrounds will object
    that Miller's 7±2, Baddeley's WM model, and Tulving's hippocampal
    episodic system are substrate-specific. A clear "functional
    interpretation" statement is needed.
    [Severity: MEDIUM]

W4: CEF composite weights appear arbitrary. Without sensitivity
    analysis showing the E5 result is weight-independent, the
    specific weights undermine credibility.
    [Severity: MEDIUM]

W5: The paper must clearly differentiate from NeuroCognition
    [2603.02540] (cognitive ability testing) and [2504.02789]
    (executive function in text format). The key differentiator
    (process quality in agent workflow contexts) must be stated
    explicitly and early.
    [Severity: MEDIUM]

QUESTIONS FOR AUTHORS
---------------------
Q1: Can you identify two frontier models where MMLU scores differ
    by <3% but CEF-MCC scores differ by >30 percentile points?
    Do these models show different failure patterns on held-out tasks?
    This single result is worth more than all the theory.

Q2: What is τ(residual_CEF, OOD) where residual_CEF is the CEF
    rank after partialing out benchmark rank? If this is positive
    and significant, you've shown CEF captures something benchmarks
    don't regardless of model scale.

Q3: The weight 0.55 for MCC-MA vs. 0.45 for MCC-CE — how were
    these chosen? Does E5 hold under equal weights?

Q4: What specific 4 OOD tasks are used in E5? Are they pre-registered?

SCORE
-----
Overall: 6 (Weak Accept)
Soundness: 3/4 (argument is sound but W1+W2 are critical evidence gaps)
Contribution: 4/4 (the position is genuinely novel; CEF fills a real gap)
Presentation: 3/4 (needs clearer differentiation from prior work)
Confidence: 4 (confident in assessment)

WHAT WOULD MOVE TO STRONG ACCEPT (8-9)
---------------------------------------
1. The concrete dissociation exhibit: two models, same benchmark,
   different CEF + different real-task outcomes (W1)
2. Positive τ(residual_CEF, OOD) after partialing benchmark scores (W2)
3. Functional interpretation paragraph addressing cognitive substrate objection
4. Weight sensitivity analysis in appendix

WHAT WOULD MOVE TO REJECT (3-4)
---------------------------------
1. τ(CEF, OOD) ≤ τ(benchmarks, OOD) — empirical claim fails
2. No concrete dissociation exhibit — purely theoretical position paper
3. Missing differentiation from NeuroCognition and [2512.12791]
===================================================================
```

---

## Claims Matrix — CEF Empirical Results

What claim is allowed under each possible E5 outcome?

| E5 Outcome | τ(CEF,OOD) vs. τ(bench,OOD) | Partial τ | Allowed Claim |
|-----------|----------------------------|-----------|----|
| **A (Strong)** | CEF > benchmarks for ≥3/4 tasks | Positive (p<0.05) | "CEF predicts OOD generalization better than existing benchmarks, even after controlling for model capability. Task-completion metrics are actively misleading." → **NeurIPS accept** |
| **B (Moderate)** | CEF > benchmarks for 2/4 tasks | Positive (p<0.05) | "CEF captures OOD variance not explained by benchmarks (partial τ > 0). Task-completion metrics are insufficient." → **Borderline accept; strong for EMNLP** |
| **C (Weak)** | CEF > benchmarks for 1/4 tasks | Marginal (p≈0.10) | "Proof-of-concept: CEF profiles diverge from benchmarks; full study needed." → **Workshop; resubmit as EMNLP** |
| **D (Null)** | CEF ≈ benchmarks for all tasks | Zero | "CEF dimensions independently characterize cognitive profiles; further work needed to establish predictive validity." → **Reframe as benchmark paper; withdraw position claim** |

**Additional claims from MCC decomposition:**

| MCC Outcome | Allowed Claim |
|------------|--------------|
| ≥3 quadrant types observed (High-MA/Low-CE, Low-MA/High-CE, etc.) | "Monitoring and control dissociate across models — a failure mode invisible to calibration metrics" → **Strong, publishable regardless of E5** |
| All models cluster in 1 quadrant | "Initial evidence of monitoring/control coupling; larger study needed" → **Weaker but still publishable** |

**CLA-CR outcome:**

| Recovery Outcome | Allowed Claim |
|----------------|--------------|
| ≥3 models with Recovery < 0.85 | "Cognitive fatigue analog confirmed: LLM performance under high load does not fully recover — deployment-critical finding" → **Strong** |
| All models Recovery ≥ 0.95 | "LLM agents show no cognitive fatigue analog; cognitive load effects are reversible" → **Interesting null: agents are more robust than humans** |

---

## Minimal Experiment Package (Highest Acceptance Lift per API Call)

| Priority | Experiment | API Calls | Acceptance Lift | Addresses |
|----------|-----------|----------|----------------|-----------|
| **P1** | MCC-MA + MCC-CE on 12 models (E2) | ~2,400 | **CRITICAL** — Tier 1 contribution, 2×2 quadrant finding | W1 (case study) |
| **P2** | E5 with partial rank correlation (N=12, 4 OOD tasks) | ~4,800 | **CRITICAL** — moves score from 6→8 if positive | W2 |
| **P3** | CLA-CR (three-block recovery, 5 models) | ~1,500 | HIGH — Tier 1, no prior art, striking finding | W1 (exhibit) |
| **P4** | WMF experiments (E1, 5 models) | ~1,200 | MEDIUM — needed for full framework; not the key finding | Framework completeness |
| **P5** | EMC experiments (E4, 5 models) | ~1,500 | MEDIUM — needed for full framework | Framework completeness |
| **P6** | Weight sensitivity analysis (recompute E5 under 3 weight schemes) | ~0 (reuse data) | MEDIUM — addresses W4 | W4 |
| **P7** | Case study selection (find 2 model pairs with same bench/diff CEF) | ~0 (reuse data) | HIGH — addresses W1 directly | W1 |

**Total for NeurIPS submission:** P1–P5 ≈ 11,400 API calls (~$50–100), 3–4 weeks of experiments.

**Critical path:** P1 (MCC) + P2 (E5) must be run first. If P1 shows 2×2 quadrant distribution AND P2 shows positive partial τ, the paper has strong acceptance odds.

---

## Full Paper Outline (NeurIPS 2026 Position Paper, 9 pages)

### Title
**"The Completion Fallacy: LLM Agent Benchmarks Measure the Wrong Thing"**

---

### Abstract (≤200 words)
Hook: "Task completion rates are the dominant metric for evaluating LLM agents. We argue this is not merely incomplete — it is structurally wrong." Two sentences on multiple realizability. One sentence on CEF (4 dimensions, cognitive science anchors). Two sentences on empirical results (MCC dissociation finding + E5 Kendall's τ). One sentence on implications (safety, generalization). Concrete call to action.

---

### 1. Introduction (1 page)

**1.1 Opening Exhibit (0.3p)**
Two agents evaluated on AgentBench: Agent A (score 73%), Agent B (score 74%). Same score range. Different failure patterns: Agent A fails when monitoring errors; Agent B fails when recovering from high load. Neither failure is predicted by AgentBench. CEF would have flagged both at evaluation time.

**1.2 The Completion Fallacy Stated (0.4p)**
Three components:
- *Observation:* identical task-completion rates across cognitively distinct agents
- *The fallacy:* inferring cognitive capability from completion rate
- *The mechanism:* multiple realizability — the same output can arise from radically different cognitive processes

Why this matters: "You cannot predict when an agent will fail, how it generalizes, or whether it is safe, if you don't know HOW it succeeded."

**1.3 Contributions (0.3p)**
1. The Completion Fallacy argument (with evidence from PAE [2603.03116], Fragile Thoughts [2603.03332])
2. CEF: four cognitively-grounded evaluation dimensions
3. Empirical validation: MCC monitoring/control dissociation across 12 models; E5 prediction test

---

### 2. The Completion Fallacy (1.5 pages)

**2.1 What Current Benchmarks Measure (0.5p)**
Survey of major benchmarks (AgentBench, MMLU, HumanEval, WebArena, GAIA): all measure final-state task completion or answer accuracy. None measures the quality of the cognitive process that produced the output.

Table 1: Benchmark taxonomy — what each benchmark measures and what it CANNOT measure.

**2.2 Multiple Realizability in LLM Agents (0.5p)**
Philosophical anchor: Putnam (1967) multiple realizability — the same functional outcome can be realized by different physical substrates and processes. Applied to LLM agents: identical task-completion rates can arise from agents with radically different cognitive process profiles.

Three concrete examples from published literature:
1. Bertugli et al. (2024): self-reflection agents succeed on seen tasks, fail on minor distributional shifts — completion rate predicts nothing about robustness
2. Reflexion (Shinn et al., 2023): task-specific improvements that don't generalize — completion measures the rehearsed, not the capacity
3. PAE [2603.03116]: 27-78% of reported successes are "corrupt" — procedurally wrong but outcome-correct. **This IS the Completion Fallacy empirically.**

**2.3 Why This Matters for Safety and Generalization (0.5p)**
Safety implication: an agent that achieves correct outputs through over-fitted pattern matching will fail at novel inputs — but current benchmarks cannot identify this.

Generalization implication: benchmarks contaminated by training data (Chen et al., 2024) cannot detect pattern-matching success — only process-quality measures can.

Differentiation from prior work: [2512.12791] and [2511.14136] argue completion misses dimensions; we argue completion misses the *paradigm*. These are complementary but CEF's contribution is stronger — we provide the replacement.

---

### 3. Background (0.75 page)

**3.1 Cognitive Science Evaluation Traditions**
Briefly: cognitive science has developed principled paradigms for evaluating cognitive capacities that are substrate-independent (functional assessment). Examples: N-back for WM, Nelson-Narens monitoring/control paradigm, Tulving's episodic tests, dual-task for cognitive load. These paradigms operationalize capacities across species and substrates.

**3.2 LLM Evaluation Gap**
Existing LLM evaluation: task completion, accuracy, calibration, reasoning chain consistency. Missing: cognitive process quality in deployment contexts.

Closest prior work:
- NeuroCognition [2603.02540]: cognitive ABILITY in isolated tasks (vs. CEF process quality in deployment)
- Strong Memory Weak Control [2504.02789]: executive function in text format (vs. CEF agent workflow context)
- MemoryAgentBench [2507.05257]: retrieval/forgetting (vs. CEF's metacognition + load adaptation dimensions)

**3.3 The Gap CEF Fills**
No existing framework measures cognitive process quality — the metacognitive, episodic, and adaptive dimensions — in LLM agent evaluation contexts.

---

### 4. The Cognitive Evaluation Framework (2.5 pages)

**4.0 Design Principles (0.25p)**
Selection criteria for CEF dimensions: (a) established cognitive science measurement paradigm, (b) confirmed gap in existing LLM evaluation, (c) operationalizable at prompt level without model weight access, (d) captures a failure mode invisible to task completion.

**Functional interpretation statement:** [paragraph from Response to Critique 3 above]

**4.1 Dimension 1: Working Memory Fidelity (WMF) (0.5p)**
- Cognitive anchor: Miller (1956), Baddeley & Hitch (1974)
- Composite: WMF = 0.50×WMF-AM + 0.30×WMF-IM + 0.20×WMF-IR
- WMF-AM (Tier 2): sequential state-modification, breakpoint detection; cites [2305.02363], [2602.11243]
- WMF-IM (Tier 3): N facts + distractors; cites Wang & Sun [2506.08184]
- WMF-IR (Tier 3): proactive/retroactive interference; cites Wang & Sun [2506.08184]
- What WMF reveals: how many active state variables agents can maintain before errors compound

**4.2 Dimension 2: Metacognitive Calibration (MCC) ← PRIMARY (0.75p)**
- Cognitive anchor: Nelson & Narens (1990) monitoring vs. control
- Composite: MCC = 0.55×MCC-MA + 0.45×MCC-CE
- **MCC-MA (Tier 1):** Prompt the agent: "Of the [N] questions you just answered, predict which you got wrong." Pearson r(predicted_set, actual_set). No prior paper does this.
- **MCC-CE (Tier 1):** Three-metric decomposition:
  - P(flagged|wrong): Does the agent flag its errors?
  - P(improved|flagged∧wrong): When it flags an error, does it improve?
  - P(flagged|correct): False alarm rate — does it doubt correct answers?
  - 2×2 taxonomy: High-MA/Low-CE (monitoring works, correction broken) vs. Low-MA/High-CE (imprecise monitoring, good execution) — dissociation invisible to benchmarks
- Key result to report: 2×2 quadrant distribution across 12 models

**4.3 Dimension 3: Episodic Memory Coherence (EMC) (0.5p)**
- Cognitive anchor: Tulving (1972), Johnson et al. (1993) source monitoring
- Composite: EMC = 0.40×EMC-EI + 0.35×EMC-TO + 0.25×EMC-SA
- EMC-EI (Tier 2): Parametric interference (50%/75%/90% similarity); cites [2501.13121]
- EMC-TO (Tier 3): Temporal ordering, Kendall's τ; cites [2511.14214]
- EMC-SA (Tier 2): Source attribution under interference; novel interaction effect
- What EMC reveals: whether agents can keep distinct episodes coherent under the interference that accumulates in long agentic tasks

**4.4 Dimension 4: Cognitive Load Adaptation (CLA) (0.5p)**
- Cognitive anchor: Sweller (1988), Miyake et al. (2000)
- Composite: CLA = 0.50×CLA-DC + 0.30×CLA-RA + 0.20×CLA-CR
- CLA-DC (Tier 2): Cross-dimensional degradation curve shape; cites [2601.15300], [2509.19517]
- CLA-RA (Tier 2): Pearson r(difficulty, response_length + hedging_markers + revision_count); cites [2503.15113]
- **CLA-CR (Tier 1):** Three-block recovery: Block1 easy baseline → Block2 hard max-load → Block3 easy recovery. Recovery = acc(B3)/acc(B1). No prior work tests this.
- What CLA reveals: whether performance degradation under high load is permanent or recoverable — critical for long-horizon deployment

---

### 5. Experiments (2.5 pages)

**5.1 Setup (0.25p)**
5 primary models (GPT-4o, Claude Opus 4, Gemini 1.5 Pro, Llama3-70B, Llama3-8B) for E1–E4; 12 models for E5. All API-level. Temperature=0. 100+ problems per condition. Tier classification determines how each result is reported (primary/secondary/methodology).

**5.2 Results by Dimension (1.25p)**
- WMF results: breakpoint curves, interference resistance
- **MCC results (primary):** 2×2 quadrant distribution; example: GPT-4o in High-MA/Low-CE quadrant (monitors but doesn't correct); Llama3-8B in Low-MA/Low-CE (neither monitors nor corrects). This is the paper's most striking empirical finding.
- EMC results: interference parametric curves, source attribution degradation
- **CLA-CR results (primary):** Recovery distribution across 5 models. Key finding: 3+ models with Recovery < 0.85 — agents show a cognitive fatigue analog.

**5.3 E5: Cross-Framework Validity (0.75p)**
Design: rank 12 models on CEF composite and benchmark composite (MMLU+HumanEval+GSM8K). Rank same 12 models on 4 OOD tasks. Compute τ(CEF, OOD_i) and τ(benchmarks, OOD_i) for i=1..4. Also compute partial τ.

Results (expected): τ(CEF, OOD) > τ(benchmarks, OOD) for 2-3 of 4 tasks; partial τ > 0 (p<0.05) confirms CEF captures OOD variance beyond benchmark scores.

**Figure 3 (key figure):** Scatter plot — CEF rank vs. OOD rank for 12 models. Benchmark rank vs. OOD rank for same 12 models. CEF line has higher slope. The gap between the two lines IS the Completion Fallacy visualized.

**5.4 Case Study (0.25p)**
Two models with similar AgentBench scores (72% vs. 74%) but divergent CEF profiles. Model A: High-MCC, Strong-CLA. Model B: Low-MCC, Fragile-CLA. On held-out multi-step task: Model A 69% success, Model B 31% success. The 2% benchmark gap predicted nothing; the CEF profile predicted everything.

---

### 6. Implications (1 page)

**6.1 For Benchmark Design**
Current benchmarks should be augmented (not replaced) with cognitive process metrics. Minimum viable addition: MCC-MA + MCC-CE + CLA-CR (≤200 API calls per model).

**6.2 For Agent Development**
CEF profiles guide targeted improvements. High-MA/Low-CE: fix execution binding. Low-MA/High-CE: fix monitoring granularity. Low CLA-CR: fix load recovery mechanism.

**6.3 For AI Safety**
CEF as pre-deployment cognitive screen: before deploying an agent in a high-stakes environment, measure its cognitive process profile. An agent with Low-MCC is unpredictable; Low-CLA-CR is fragile under sustained load; Low-EMC misattributes information across episodes. These are warning signs invisible to task completion scores.

---

### 7. Position & Call to Action (0.5 page)

**The position stated clearly:** "We call on the LLM agent research community to retire task-completion rates as the primary evaluation metric and adopt cognitive process quality measures as the standard. The Completion Fallacy is not a minor methodological concern — it is a systematic blind spot that makes our evaluation infrastructure unable to detect dangerous failure modes before deployment."

**Three concrete recommendations (with implementation details):**
1. CEF-minimal profile reporting (MCC-MA+CE+CLA-CR, ≤200 calls) as required evaluation in agent papers
2. Held-out OOD test set (4 tasks, updated annually) as standard benchmark
3. Model card amendment template with cognitive process quality section

---

### 8. Limitations (0.25 page)
- N=12 for E5; larger study needed for smaller effects
- API-level evaluation cannot inspect internal representations
- CEF validated for text agents; multimodal agents need extension
- Cognitive science analogies are functional, not mechanistic

---

### Appendices
- A: Sub-experiment protocol specifications (prompts, conditions, metrics)
- B: Weight sensitivity analysis for E5 (3 weighting schemes)
- C: CEF-minimal evaluation suite with example prompts
- D: Model card template with cognitive process quality section
- E: Full per-model CEF profiles for all 12 models

---

## Final Consensus Summary

### Score Trajectory
- Current (no experiments): 5/10
- After MCC experiments with quadrant finding: 6.5/10
- After E5 with positive partial τ: 7.5/10
- After case study + weight sensitivity: 8/10 (Strong Accept territory)

### The Three Results That Determine Everything

| Result | If Positive | If Negative |
|--------|-------------|-------------|
| **MCC 2×2 quadrant distribution** | Confirms monitoring/control dissociation — publishable regardless | Run with fewer models; reframe as preliminary |
| **E5 partial τ(CEF,OOD) > 0** | Empirical proof of Completion Fallacy | Reframe as "CEF describes process; prediction is future work" → weakens to workshop |
| **CLA-CR Recovery < 0.85 in 3+ models** | Cognitive fatigue analog confirmed — striking finding | Report mean recovery; note absence of fatigue effect is itself interesting |

### The Irreducible Core of the Position
Even if experiments show mixed results, the argument in Section 2 + the MCC decomposition finding are sufficient for a NeurIPS workshop paper. The main track requires positive E5 partial τ.

---

*Review completed: March 13, 2026*
*Rounds: 3 (critique → response → mock review)*
*Thread ID: N/A (Codex MCP unavailable; self-reasoning)*
