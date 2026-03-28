# Pre-Specified Analysis Plan — CEF WMF-AM Study

**Document type:** Frozen prospective analysis plan (GitHub-committed)
**Note:** This is a GitHub-committed specification document, not a registry-based preregistration
(OSF/AsPredicted). The commit timestamp serves as evidence of prior specification, but does not
provide the same guarantees as a formal registry preregistration.

---

## Study Title
The Completion Fallacy: A Cumulative Arithmetic State-Tracking Probe Predicts LLM Agent Performance

## Date Frozen
This document was committed to the repository before collection of the N=20 expansion set data.
Git commit hash: see repository history.

---

## Pre-Specified Primary Hypotheses

**H1 (Predictive Validity):**
WMF-AM (4-seed mean, K=3/5/7 arithmetic state-tracking probe) will significantly predict
performance on the 10-task deterministic agent battery (Kendall's τ > 0, one-sided;
two-sided α = 0.025 after Bonferroni correction for 2 hypotheses).

**H2 (Template Stability):**
Model rankings under the "bare" prompt template will significantly correlate with rankings
under the "chat" template (Kendall's τ > 0, two-sided α = 0.025 after Bonferroni correction).

---

## Pre-Specified Model Set (Frozen Before Expansion Data Collection)

**N = 20 models, 13 architectural families:**

Original 7:
- deepseek-r1:14b (DeepSeek)
- qwen2.5:32b (Qwen)
- qwen2.5:14b (Qwen)
- gemma2:27b (Gemma)
- qwen2.5:7b (Qwen)
- mistral:7b (Mistral)
- llama3.1:8b (Llama)

Expansion 8:
- phi3:14b (Phi)
- gemma2:9b (Gemma)
- qwen2.5:3b (Qwen)
- llama3.2:3b (Llama)
- deepseek-r1:7b (DeepSeek)
- mixtral:8x7b (Mistral)
- command-r:35b (Cohere)
- yi:34b (Yi)

Small baseline 5:
- gemma2:2b (Gemma)
- qwen2.5:1.5b (Qwen)
- tinyllama:1.1b (TinyLlama)
- llama3.2:1b (Llama)
- qwen2.5:0.5b (Qwen)

**Exclusion rule:** No post-hoc model removal. All 20 models included in the primary analysis
regardless of performance level.

---

## Pre-Specified Primary Test Statistic

- Kendall's τ-b (handles ties)
- Two-tailed p-value
- Bonferroni correction: α = 0.025 per hypothesis (2 hypotheses total)
- Bootstrap 95% CI: 10,000 nonparametric model-level resamples (percentile method)

---

## Pre-Specified Downstream Battery

10-task deterministic agent battery (released at https://github.com/dengzhehou/CEF-battery):
- T1: State-tracking (3 tasks: multi-step inventory, sequential booking, resource allocation)
- T2: Multi-step reasoning (3 tasks: deductive inference, algebraic word problems, constraint propagation)
- T3: General problem-solving (2 tasks: combinatorial planning, conditional execution)
- T4: Constrained planning (2 tasks: scheduling, resource allocation under ordering)

Scoring: deterministic exact-match verification (no LLM judge).

---

## All Other Analyses Are Exploratory

The following analyses were NOT pre-specified and should be treated as hypothesis-generating only:
- Partial τ analyses (controlling for completion score or model scale)
- Within-family analyses
- Leave-one-out / leave-family-out robustness
- WMF/non-WMF task split analyses
- Convergent/divergent validity analyses
- Any comparison to Minerva Quantity State

---

## Limitations of This Document

1. This is a GitHub-committed document, not a registry-based preregistration.
   It does not provide the auditability guarantees of OSF or AsPredicted.
2. The original 7-model pilot was run before this document; only the expansion set
   (N=8 expansion + N=5 small models) was collected after this specification was frozen.
3. The Bonferroni correction covers 2 hypotheses; additional comparisons discovered
   in exploratory analyses do not receive this protection.
