# Novelty Check Report — CEF / WMF-AM (Round 3, 2026-03-22)

## Proposed Method
A calibrated, ablation-backed probe (WMF-AM) for cumulative arithmetic state tracking under load, validated via pre-specified rank-correlation analyses (Kendall's τ) against a downstream deterministic agent battery across 20 open-weight models (13 families, 0.5B–35B).

---

## Core Claims

| # | Claim | Novelty | Closest Prior Work |
|---|-------|---------|-------------------|
| 1 | Completion Fallacy framing | **LOW** | PAE (2603.03116), AgentProcessBench (2603.14465) |
| 2 | WMF-AM probe + ablation suite | **MEDIUM** | Minerva QS (2502.03358) + state-tracking work (2511.10457, 2503.02854) |
| 3 | K-calibration / discriminability finding | **MEDIUM** | Benchmark² (2601.03986) |
| 4 | Predictive validity (probe → agent battery, τ=0.612) | **HIGH** | AgentProcessBench (tangential) |
| 5 | Process-outcome dissociation quantification | **MEDIUM** | PAE (2603.03116), Robust Answers Fragile Logic (2505.17406) |

---

## Closest Prior Work

| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| PAE (arXiv:2603.03116) | 2026 | arXiv | Corrupt success, trajectory evaluation | Already cited; trajectory-level, not controlled probe |
| Minerva QS (arXiv:2502.03358) | 2025 | ICML | K-parameterized quantity state task | No ablations, no predictive validity, K=200 floors all |
| AgentProcessBench (arXiv:2603.14465) | 2026 | arXiv | Step-level process quality | Human-annotated trajectories, no controlled probe, no predictive validity |
| Exploring State Tracking (arXiv:2511.10457) | 2025 | arXiv | LLM state tracking benchmark | Entity/object tracking, no arithmetic accumulation, no ablations |
| (How) Do LLMs Track State (arXiv:2503.02854) | 2025 | arXiv | Mechanistic state-tracking study | Permutation composition, no downstream agent validation |
| Benchmark² (arXiv:2601.03986) | 2026 | arXiv | Kendall's τ across benchmarks | Peer benchmark consistency ≠ probe→agent predictive validity |
| Probing Arithmetic Errors (arXiv:2507.12379) | 2025 | EMNLP | Probing for arithmetic errors | Single-step error detection, not cumulative sequential accumulation |
| Scaling Laws State Dynamics (arXiv:2505.14892) | 2025 | arXiv | LLM state dynamics scaling | Box/DFA tracking, no arithmetic, no downstream validation |

---

## External Reviewer Assessment (GPT-5.4 xhigh)

**Overall Novelty Score: 6.5/10**
**Recommendation: PROCEED WITH CAUTION**

The genuinely novel core is the **combination** of:
- Calibrated narrow probe in no-scratchpad regime
- Targeted ablation suite (yoked cancellation, K=1, non-arithmetic ceiling)
- Validated predictive rank-correlation against a downstream agent battery

**Key warning from reviewer:**
> "Do NOT sell Claim 1 as the novelty. Reviewers will reject that positioning immediately. By March 2026, 'completion is insufficient' is already an established complaint, not a new framing."

---

## Actions Taken

### Citations Added to paper
**Must-add (added):**
- `fan2026agentprocessbench` (arXiv:2603.14465) — in empirical support paragraph
- `xu2025robustanswers` (arXiv:2510.13272) — in empirical support paragraph
- `rezaee2025statetracking` (arXiv:2511.10457) — in WMF-AM related work
- `li2025howtrack` (arXiv:2503.02854) — in WMF-AM related work
- `scaling2025statedynamics` (arXiv:2505.14892) — in WMF-AM related work
- `zhou2026benchmark2` (arXiv:2601.03986) — in K-calibration section

**Did NOT add (out of scope or low priority):**
- `akshathala2025beyond` (arXiv:2512.12791) — PAE and AgentProcessBench already cover this angle sufficiently
- `kuang2025enconda` (arXiv:2510.25694) — software-agent specific, too narrow
- `sun2025probing` (arXiv:2507.12379) — single-step arithmetic probing, too different

### Strategic notes
- The paper's Completion Fallacy framing is now well-motivated by new citations, but **the abstract/intro already correctly positions predictive validity as the primary contribution**
- The paper is positioned correctly as of the latest edits (predictive validity leads, Completion Fallacy is background motivation)

---

## Suggested Positioning (from reviewer)

**Sell the paper as:** *"A validated process probe for agent capability, with calibrated difficulty and targeted ablations."*

Not: "We name a new phenomenon (completion fallacy)."
Not: "LLMs can't track state" (already known).

The K-calibration vs. Minerva finding and the probe-to-agent τ=0.612 are the twin novelty pillars.
