# Paper Improvement Log — Research Paper Conversion Loop

**Loop type:** Research paper (position paper → main track research paper)
**Reviewer:** GPT-5.4 xhigh (thread: 019d09a5-f92c-7ff2-b6c8-0c5b19780c4f)
**Date:** 2026-03-20

## Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Round 0 (baseline after research reframe) | 5/10 | Almost | Baseline: claim-evidence mismatch, external validity, stats clarity |
| Round 1 | 6/10 | Yes | Narrowed claims, added robustness, clarified confirmatory/exploratory |
| Round 2 | 6/10 | Yes | Final polish: title, construct scope, cluster bootstrap |

## Round 1 Review & Fixes

<details>
<summary>GPT-5.4 xhigh Review (Round 1)</summary>

**Overall Score:** 5/10

**Summary**
The paper introduces WMF-AM, a probe for cumulative arithmetic state tracking under load, and reports that across 20 open-weight models it correlates strongly with a separate 10-task agent battery, including after partial controls for a custom correctness benchmark and model size. The core empirical result is interesting and the ablation suite is better than average, but the current evidence supports a narrower claim than the paper makes: this is a promising diagnostic on a limited model set, not yet a strong demonstration of a general "Completion Fallacy" or a broad predictor of LLM agent performance.

**Strengths**
- Strong central signal on an independent downstream criterion: τ=0.612 on N=20 is meaningful, and the paper does try to show incremental validity beyond completion and scale.
- Good negative controls and ablations: the K=1 control, yoked cancellation, and non-arithmetic assignment tracking meaningfully test superficial alternative explanations.
- Robustness checks are thoughtful: seeds, prompt wrappers, paraphrases, and leave-one-out analyses all help reduce concern that the result is a single-template artifact.
- The paper is unusually explicit about scope and limitations for this genre.
- The problem itself is important: process-sensitive evaluation for agents is a worthwhile direction.

**Weaknesses**
1. CRITICAL: Claim-evidence mismatch. Title, abstract, and "Completion Fallacy" framing broader than what is actually shown.
2. CRITICAL: External validity and effective sample size limited. N=20 with family clustering (6 Qwen, 3 DeepSeek), range restriction, no closed/API models.
3. MAJOR: Construct validity incomplete. WMF-AM ablations don't establish broader "working-memory fidelity" construct.
4. MAJOR: "Beyond completion" comparison weaker than framing implies. Custom 100-item battery, not standard benchmarks.
5. MAJOR: Downstream agent criterion too small/underspecified. 10-task deterministic battery.
6. MAJOR: Statistical reporting needs tightening. Confirmatory vs exploratory unclear.
7. MINOR: Over-extended by broader CEF taxonomy.
8. MINOR: Writing clarity and self-containedness issues.

**Verdict:** Almost

</details>

### Round 1 Fixes Implemented
1. Title narrowed: "A Cumulative Arithmetic State-Tracking Probe Predicts Open-Weight LLM Agent Performance"
2. Abstract softened: added "in this open-weight model sample" qualifier; "systematically miss" → "miss" in sample
3. Introduction scope paragraph: explicitly states what is and is not established; Completion Fallacy framing is "motivation" not "universal law"
4. Confirmatory/exploratory distinction made explicit in stats section
5. Leave-one-family-out robustness added: τ ranges 0.503–0.649 across families
6. Agent battery expanded: full task category descriptions + leave-one-category-out (τ=0.549–0.623)
7. "Beyond completion" transparency: explicit caveat that control is custom 100-item battery; log(params) as more theory-neutral proxy
8. Discussion limitations strengthened with explicit caveat about standard baseline comparison
9. Overfull hbox (1.08pt) fixed with {\sloppy}

---

## Round 2 Review & Fixes

<details>
<summary>GPT-5.4 xhigh Review (Round 2)</summary>

**Overall Score:** 6/10

**Summary**
This revision materially improves the paper. It now reads as a focused empirical validation that a cumulative state-tracking probe has incremental predictive value for ranking 20 open-weight models on a custom deterministic agent battery, rather than as a broad claim about LLM agents in general. The core result is interesting and the paper is now much better calibrated, but the evidence is still narrow enough that generality remains the main limitation.

**Strengths**
- Stronger claim-evidence alignment.
- Much better empirical transparency.
- Robustness story substantially improved.
- Solid core ablation package.
- Contribution now clearer.

**Weaknesses**
- MAJOR: External validity still limited (20 open-weight models, no API validation).
- MAJOR: Downstream criterion still small and custom (10-task battery).
- MAJOR: Construct validity still partial.
- MINOR: "Incremental prediction" in title rests on exploratory partial-τ.
- MINOR: CEF taxonomy extends beyond strongest evidence.

**Actionable Fixes**
- External validity: add held-out API model set or cluster-aware inference.
- Agent criterion: full task prompts/scoring in appendix, per-task correlations.
- Construct validity: tighten terminology to "cumulative arithmetic state tracking under load" throughout.

**Verdict:** Yes — "ready for submission as a scoped empirical paper"

</details>

### Round 2 Fixes Implemented
1. Title updated to specify "Arithmetic State-Tracking" and "Open-Weight"
2. Family-clustered bootstrap added to stats methodology
3. Construct scope caveat added in Section 3 opening: "current evidence supports the narrower construct of cumulative arithmetic state tracking under load, not broad working-memory capacity"
4. Overfull hbox (1.42pt) in stats section fixed with {\sloppy}

---

## PDFs
- `cef_paper_v8_research_round0_original.pdf` — After all position→research paper conversions (baseline)
- `cef_paper_v8_research_round1.pdf` — After Round 1 fixes
- `cef_paper_v8_research_round2.pdf` — Final version (= cef_paper_v7.pdf)

## Remaining Issues (require new experiments)
- External validity: no closed/API model evaluation (GPT-4o, Claude, Gemini)
- Agent battery: only 10 tasks; second downstream criterion would strengthen
- Standard benchmark comparison: MMLU/GSM8K not re-evaluated in-study (only model card values)
- Factor analysis: N=20 insufficient; needs N≥30 for dimensional structure
