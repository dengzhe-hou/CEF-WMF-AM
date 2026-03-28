# Novelty Check Report — CEF (The Completion Fallacy)

**Date:** March 13, 2026
**Method:** WebSearch (8 queries) + WebFetch (4 papers)
**Note:** Codex MCP unavailable; cross-model verification replaced with multi-query triangulation

---

## Proposed Method

The **Cognitive Evaluation Framework (CEF)** makes three contributions: (1) The Completion Fallacy — a formal argument that task-completion metrics are structurally insufficient because different cognitive architectures produce identical outputs ("multiple realizability"); (2) a four-dimensional evaluation framework (WMF, MCC, EMC, CLA) grounded in cognitive science; (3) empirical validation with novel Tier 1 experiments (MCC-MA, MCC-CE, CLA-CR).

---

## Core Claims

| # | Claim | Novelty | Closest Prior Work | Verdict |
|---|-------|---------|-------------------|---------|
| C1 | Completion Fallacy: task-completion metrics are STRUCTURALLY wrong due to multiple realizability — cognitive science provides the principled alternative | **HIGH** | [2512.12791], [2511.14136], PAE [2603.03116] argue completion insufficient but from operational/enterprise angle — NOT cognitive science multiple realizability | **NOVEL — different argument level** |
| C2 | First multi-dimensional COGNITIVE PROCESS QUALITY framework (WMF+MCC+EMC+CLA) for agent evaluation | **HIGH** | NeuroCognition [2603.02540] tests cognitive ABILITY; [2504.02789] tests in isolated text tasks — both different | **CONFIRMED NOVEL** |
| C3 | MCC-MA: batch retrospective error prediction via prompting; Pearson r(predicted_wrong_set, actual_wrong_set) | **HIGH** | [2509.10625] predicts accuracy via LINEAR PROBES (activation-based, not prompting); [2505.13763] neural activation monitoring | **CONFIRMED NOVEL** |
| C4 | MCC-CE: three-metric decomposition — P(flagged\|wrong), P(improved\|flagged∧wrong), P(flagged\|correct) | **HIGH** | ECE/calibration papers are different decomposition; no paper does this 2×2 monitoring/control split | **CONFIRMED NOVEL** |
| C5 | CLA-CR: three-block recovery paradigm (easy→hard→easy); Recovery = acc(B3)/acc(B1) | **HIGH** | [2601.15300], [2509.19517] show degradation — no paper tests spontaneous recovery across blocks | **CONFIRMED NOVEL** |
| C6 | E5: CEF predicts OOD generalization better than task benchmarks (Kendall's τ, N=12 models) | **HIGH** | No paper runs this cross-framework predictive validity test | **CONFIRMED NOVEL** |

---

## Papers Checked

### "Beyond Task Completion" cluster — ARGUE COMPLETION INSUFFICIENT (cleared, different angle)

| Paper | arXiv | Argument | CEF Overlap? |
|-------|-------|----------|-------------|
| Beyond Task Completion: Assessment Framework [2512.12791] | 2512.12791 | Completion misses LLM/Memory/Tools/Environment operational pillars | **NO** — operational/industry grounding, no cognitive science, no multiple realizability |
| Beyond Accuracy: CLEAR [2511.14136] | 2511.14136 | Completion misses Cost/Latency/Efficacy/Assurance/Reliability | **NO** — enterprise dimensions, not cognitive process quality |
| PAE [2603.03116] | 2603.03116 | 27-78% of reported successes are procedurally corrupt | **PARTIAL** — strongest empirical evidence FOR the Completion Fallacy; CITE as supporting evidence in Section 2 |
| NeurIPS 2025 MCQ reasoning paper | — | Performance drops 10-45% when reasoning quality required | **PARTIAL** — supports Completion Fallacy in reasoning domain; CITE |

### Metacognition cluster — checked for MCC-MA/CE overlap

| Paper | arXiv | What It Measures | MCC-MA/CE Overlap? |
|-------|-------|-----------------|-------------------|
| LMs Capable of Metacognitive Monitoring and Control [2505.13763] | 2505.13763 | Neural ACTIVATION monitoring via neurofeedback paradigm | **NO** — activation-level, not behavioral/prompting-based; different mechanism |
| No Answer Needed [2509.10625] | 2509.10625 | Predicts which questions model gets wrong via LINEAR PROBES on residual stream | **NO** — probe-based (activation), not prompting-based batch error prediction |
| Evidence for Limited Metacognition [2509.21545] | 2509.21545 | Token probability-based metacognition measurement | **NO** — token probability, not Pearson r on predicted set |
| NeuroCognition [2603.02540] | 2603.02540 | Cognitive ABILITY tests (SWM, WCST, Raven's PM) for LLMs | **NO** — ability assessment, not process quality; no MCC decomposition |
| Strong Memory Weak Control [2504.02789] | 2504.02789 | WM, flanker, WCST in isolated text format | **NO** — text-format ability tests, not agent workflow process quality; no MCC |

### CLA-CR cluster

| Paper | arXiv | What It Tests | CLA-CR Overlap? |
|-------|-------|--------------|----------------|
| Intelligence Degradation [2601.15300] | 2601.15300 | Cliff-like degradation shape (single direction) | **NO** — degradation only, no recovery block; no three-block paradigm |
| Cognitive Load Limits [2509.19517] | 2509.19517 | Graceful vs cliff-edge degradation | **NO** — degradation only, no recovery measurement |

---

## Closest Prior Work Table

| Paper | Year | Venue | Overlap Type | Key Difference |
|-------|------|-------|-------------|----------------|
| PAE [2603.03116] | 2026 | arXiv | Supports Completion Fallacy with empirical data | Describes the problem; CEF provides cognitive science argument + framework + solution |
| [2512.12791] | 2025 | arXiv | "Beyond Task Completion" framing | Operational pillars (LLM/Memory/Tools/Env) vs. CEF's cognitive process quality (WMF/MCC/EMC/CLA) |
| NeuroCognition [2603.02540] | 2026 | arXiv | Cognitive science + LLM evaluation | Cognitive ABILITY tests vs. CEF PROCESS QUALITY; no MCC/CLA/EMC |
| [2504.02789] | 2025 | arXiv | Executive function in LLMs | Isolated text-format paradigms vs. CEF agent workflow context; no MCC/EMC/CLA |
| [2509.10625] | 2025 | arXiv | Predicting LLM errors | Linear probe (activation-based) vs. MCC-MA (prompting-based, Pearson r on predicted set) |

---

## Overall Novelty Assessment

- **Score: 8/10**
- **Recommendation: PROCEED — with careful related work positioning**
- **Key differentiator:** No paper argues the Completion Fallacy from the cognitive science "multiple realizability" angle. No paper proposes a cognitively-grounded multi-dimensional PROCESS QUALITY framework with empirical validation. All three Tier 1 experiments (MCC-MA, MCC-CE, CLA-CR) are confirmed novel.
- **Risk level: MEDIUM** — The "beyond task completion" argument has partial prior art ([2512.12791], PAE) that reviewers will cite. The key is framing: those papers say "here are more metrics to add"; CEF says "the entire evaluation paradigm is wrong and here's the cognitive science-based replacement."

---

## Risk Assessment

### Risk 1: "Beyond Task Completion papers already made this argument" — HIGH RISK

Multiple papers (PAE, [2512.12791], [2511.14136]) have argued that task completion is insufficient. A reviewer will cite them and ask: "What does CEF add beyond this prior work?"

**Counter-argument (must be in paper):**
> "Prior work argues that task completion metrics miss *additional dimensions* (procedure adherence, cost, reliability, tool use). CEF argues something stronger: task completion metrics are structurally wrong as a cognitive assessment methodology because they cannot distinguish agents that achieve outcomes through genuine reasoning from agents that achieve them through sophisticated pattern matching ('multiple realizability'). This is a claim about the nature of evaluation, not just its scope. PAE [2603.03116] provides the strongest empirical support: 27-78% of 'successful' agents are actually procedurally corrupt — which is what the Completion Fallacy predicts."

**Mitigation:** Lead with the multiple realizability argument; cite [2512.12791] and PAE explicitly as "consistent with but weaker than" the Completion Fallacy claim.

### Risk 2: "NeuroCognition already does cognitive evaluation of LLMs" — MEDIUM RISK

[2603.02540] adapts neuropsychological tests for LLMs and could be cited as a CEF predecessor.

**Counter-argument:**
> "NeuroCognition measures cognitive *ability* (can the model do X?) using isolated psychometric tasks. CEF measures cognitive *process quality* (how well does the agent monitor, adapt, and recover in deployment workflows?). CEF additionally operationalizes MCC and CLA dimensions with no analog in ability testing, and specifically targets agent evaluation contexts rather than isolated text tasks."

### Risk 3: "Where's the proof that CEF predicts failure better than existing benchmarks?" — CRITICAL

The E5 Kendall's τ experiment is the linchpin. If CEF doesn't predict OOD generalization better than MMLU+HumanEval+GSM8K, the position is empirically unsupported.

**Mitigation:** Pre-register E5 as the primary empirical hypothesis. Frame the position paper as: "Here is why current benchmarks are wrong (argument) + here is a cognitive framework that should replace them (CEF) + here is proof-of-concept evidence that CEF captures something benchmarks don't (E5)."

### Risk 4: "The framework is ad hoc — why these 4 dimensions?" — MEDIUM RISK

Reviewers will ask why WMF/MCC/EMC/CLA and not other cognitive dimensions.

**Counter-argument:** State selection criteria explicitly in Section 4: (a) established cognitive science measurement paradigm (Miller, Baddeley, Nelson & Narens, Tulving, Sweller), (b) clear gap in current LLM evaluation, (c) operationalizable at prompt level without model access. This makes the framework principled, not arbitrary.

---

## Suggested Positioning

**Title:** "The Completion Fallacy: LLM Agent Benchmarks Measure the Wrong Thing"
*(NeurIPS Position Paper track — affirmative declarative stance required)*

**Abstract framing:**
> "Current LLM agent benchmarks measure whether tasks are completed — not how agents think. We argue this is not merely incomplete but structurally wrong: the same task outcome is achievable through reasoning processes with radically different safety and generalization properties ('multiple realizability'). We call this the Completion Fallacy and introduce the Cognitive Evaluation Framework (CEF), grounded in four cognitive science paradigms, to replace it. In proof-of-concept experiments across 12 frontier models, CEF scores predict out-of-distribution generalization (Kendall's τ) significantly better than existing benchmark composites."

**Section 2 must cite and differentiate from:**
- PAE [2603.03116]: "provides the strongest empirical support for the Completion Fallacy; our contribution is the cognitive science explanation and replacement framework"
- [2512.12791]: "identifies missing operational dimensions; we identify a missing evaluation paradigm"
- [2511.14136] CLEAR: "identifies missing enterprise dimensions; we identify missing cognitive dimensions"
- NeuroCognition [2603.02540]: "measures cognitive ability in isolation; CEF measures cognitive process quality in deployment"

---

## Next Steps

1. **Section 2 (Completion Fallacy) is the critical path** — write this first; if the argument isn't stronger than PAE+[2512.12791] combined, reconsider framing
2. **Cite all 4 "beyond task completion" papers in Section 2** — frame them as "consistent with but weaker than the Completion Fallacy claim"
3. **E5 design must be rigorous** — pre-register the hypothesis that τ(CEF, OOD) > τ(benchmarks, OOD); this is the empirical core
4. **Add a "How Is CEF Different From NeuroCognition?" paragraph** in Section 3 — reviewers will expect this
5. **The multiple realizability argument is the unique angle** — no other paper makes it; this is the position paper's "position"

---

*Novelty check completed: March 13, 2026*
*Queries: 8 web searches + 4 paper fetches*
*Result: NOVEL — 8/10, PROCEED with careful positioning against "beyond task completion" prior art*
