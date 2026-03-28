# Auto Review Log — Loop 13 (CEF v7)

**Date:** 2026-03-20  
**Reviewer model:** GPT-5.4 xhigh (via Codex MCP)  
**ThreadId:** `019d07ae-8422-7ce3-8fda-0dba0fb8cc62`  
**Starting score (Loop 12 end):** 8/10

---

## Round 1

**Score: 7/10 — Almost**

### Reviewer criticisms

CRITICAL:
- Overclaiming from WMF-AM to validated CEF 4-dim framework — fix: explicit "provisional, unvalidated taxonomy" everywhere
- Criterion contamination — non-WMF split exists but not prominent; fix: make it a labeled control with clear interpretation
- GPT-4o judge (AGENT-PQ) as primary criterion concern — fix: demote to auxiliary, foreground deterministic score

MAJOR:
- N=20 fragility (all open-weight)
- WMF-AM may primarily be arithmetic probe — fix: reframe as "cumulative arithmetic state tracking"
- Agent battery too narrow for rhetorical weight — fix: restrict language
- Complementary probes too preliminary — fix: trim to compact paragraph, label explicitly preliminary

MINOR: Formal definition doesn't distinguish statistical vs practical insufficiency; CoT ceiling ambiguous; genre confusion

### Fixes Implemented

1. **Abstract**: Added "provisional," relabeled WMF-AM as "cumulative arithmetic state tracking under load," added "unvalidated working taxonomy" for CEF 4-dim structure, added non-WMF task split to abstract
2. **Introduction**: Added explicit genre statement ("position paper with pilot study")
3. **CEF Section**: WMF-AM now described as "probe of cumulative arithmetic state tracking under load" + scope clarification paragraph
4. **Pilot Section**: "Criterion contamination control (pre-registered split)" labeled paragraph, softened "directly falsifying" → "arguing against the simplest criterion-overlap explanation"
5. **Complementary probes**: Trimmed to compact paragraph, explicitly "preliminary existence checks, not validated sub-dimensions"
6. **Limitations**: Added (iii) WMF-AM arithmetic-specificity + K=1 control in progress; added (v) AGENT-PQ explicitly auxiliary; added (viii) 4-dim structure provisional
7. **Formal definition**: Added sentence distinguishing statistical from practical insufficiency

---

## Round 2

**Score: 8/10 — Almost**

### Reviewer assessment

> "The paper is materially improved. The revised framing is now closer to what the evidence can bear... The biggest gain is not new data but better epistemic discipline."

Round 1 concerns (all 3 CRITICAL): Addressed. No CRITICALs remain.

### Remaining MAJORs (all require new experiments or broader data)
1. Empirical case rests on one probe + one narrow battery — acceptable for pilot position paper
2. WMF-AM not yet fully disentangled from serial arithmetic (K=1 control in progress — Job 400)
3. "Directly falsifying" / "rules out" language too strong → replaced with "argues against"
4. External validity: all open-weight, no frontier API models

### Additional fixes applied (Round 2 → Final)
- Replaced "directly falsifying the criterion-overlap explanation" → "arguing against the simplest criterion-overlap explanation"
- Replaced "ruling out criterion overlap as the driver" → "arguing against the simplest criterion-overlap explanation" (abstract)

---

## Final Status

**Score: 8/10 — Almost**  
Paper compiled: 254.00 KiB  

**Remaining blockers (require new experiments, not writing fixes):**
- Last-operation-only control (K=1) — running as Job 400 on c03 A100
- Cloud API models (GPT-4o, Claude 3.5, Gemini) — needs OpenRouter API
- Factor analysis — requires N≥30

**Writing is at ceiling for current evidence.** All CRITICAL and most MAJOR issues resolved. Paper ready for arXiv pending K=1 control results.

---

## Score Progression (All Loops)

| Loop | R1 | R2 | Key achievement |
|------|----|----|-----------------|
| 1 | 5 | 6 | Ceiling defense, composites exploratory |
| 2 | 6 | 7 | Construct-development framing |
| 3 | 5 | 6 | WMF-AM control + MCC-CE-v2 integrated |
| 4 | 5 | 6 | Rival explanations, evidence map |
| 5 | 6 | 7 | Scope compression |
| 6 | 6 | 6 | NeurIPS format |
| 7 | 5 | 6 | Statistical rigor |
| 8 | 5 | 6 | N=15 integration |
| 9 | 6 | 7 | Template harmonization |
| 10 | 5 | 6 | N=20 + predictive validity |
| 11 | 6 | 7 | Benchmark-conditional formalization |
| 12 | 7 | **8** | Ablations integrated, claims narrowed |
| 13 | 7 | **8** | Provisional taxonomy, arithmetic probe clarity, criterion contamination |

