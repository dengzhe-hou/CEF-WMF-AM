# CLAUDE.md — CEF NeurIPS 2026

> Auto-loaded by Claude Code. Project root: `/home/hou/Research/CEF-NeurIPS2026/`

---

## Project Goal

**NeurIPS 2026 Main Track** — Position Paper submission (~May 22–28, 2026).
arXiv preprint same day as submission.

**Title:** "The Completion Fallacy: Why Task Completion Is an Insufficient Proxy for LLM Agent Process Quality"

**Current status:** Score 8/10 YES (Loop 12, GPT-5.4 xhigh). Ready for arXiv after final polish.

---

## Directory Structure

```
CEF-NeurIPS2026/
├── paper/                  ← LaTeX source + compiled figures
│   ├── cef_paper_v7.tex    ← PRIMARY paper file (20 pages, tectonic)
│   ├── cef_refs.bib
│   ├── neurips_2026.sty
│   ├── PAPER_IMPROVEMENT_LOG.md
│   └── figures/
│       ├── fig1_dissociation.{pdf,png,svg}   ← N=20, two panels
│       ├── fig2_depth_profile.{pdf,png,svg}  ← N=15, K=3/5/7
│       ├── fig3_framework.{pdf,png,svg}      ← CEF taxonomy
│       ├── fig4_validity_heatmap.{pdf,png,svg}
│       └── gen_figures_v8.py                 ← master generator (all 4 figs)
├── code/                   ← experiment scripts
│   ├── config.py           ← model list, call_model(), RESULTS_DIR
│   ├── wmf_am_*.py         ← WMF-AM probes (control, yoked, nonarith, paraphrase, etc.)
│   ├── cef_*.py            ← CEF battery, agent validation, validity, n-expansion
│   ├── mcc_ce_v2_*.py      ← MCC-CE v2 runner + questions
│   ├── emc_lite.py         ← EMC probes
│   ├── cla_adaptation.py   ← CLA probes
│   └── analysis/
│       ├── collect_n15_data.py
│       └── cross_dimension_prediction.py
├── data/                   ← JSON result files only (no logs)
│   ├── wmf_am_nonarith_20260317T150520.json        ← ablation: non-arith (N=15, ceiling=0.98)
│   ├── wmf_am_paraphrase_20260317T152625.json      ← ablation: paraphrase stability
│   ├── wmf_am_control_ollama_*.json                ← 7 models × construct validity
│   ├── wmf_am_yoked_control_ollama_*.json          ← yoked control
│   ├── cef_completion_battery_v2_20260317T122721.json  ← N=15 completion scores
│   ├── cef_wmf_multiseed_expansion8_20260317T122849.json  ← N=15 WMF-AM
│   ├── mcc_ce_v2_results.json                      ← MCC floor effect fixed
│   ├── cef_agent_validation_all.json               ← downstream agent battery
│   ├── cef_predictive_validity.json
│   ├── cef_incremental_validity.json
│   └── nexp/                                       ← per-model N-expansion (8 models)
├── docs/
│   ├── PAPER_IMPROVEMENT_LOG.md   ← full review loop history
│   ├── AUTO_REVIEW_CEF_V7_MAINTRACK.md  ← latest review (8/10)
│   ├── NOVELTY_CHECK_CEF.md
│   └── REVIEW_CEF_DEEP.md
├── literature/             ← 58 reference PDFs
└── overleaf/
    └── cef_paper_overleaf.zip   ← upload to Overleaf directly
```

---

## Paper Status

| Loop | Score | Key Achievement |
|------|-------|----------------|
| Loop 1 | 5→6 | Ceiling defense, composites labeled exploratory |
| Loop 2 R1 | 6 | 7 new references |
| Loop 2 R2 | 7 | Construct-development framing, falsification criteria |
| Loop 12 R1-R2 | 7→**8** | Non-arith ablation integrated, narrowed agent battery claim |

**Score progression (all loops):** 5,6,6,6,5,6,5,6,6,7,5,6,5,6,6,7,6,6,5,6,5,6,6,7,5,6,6,7,7,**8**

**Remaining (from 8/10 review):**
- N=7 still underpowered → need N≥15 via OpenRouter (GPT-4o, Claude 3.5, Gemini)
- 4-dim structure empirically provisional → needs CFA/EFA at N≥15
- Standalone arithmetic control pending (last-operation-only probe)

---

## Key Experimental Results

1. **Dissociation exhibit** — same completion (0.95), divergent CEF (0.387–0.627, Δ=0.24)
2. **WMF-AM control task** — 5/7 models near-ceiling on inert-op control → validates AM measures arithmetic accumulation, not just entity tracking
3. **Non-arithmetic ablation** — 15 models × 3 domains, mean accuracy 0.98 → ceiling confirms arithmetic accumulation is the hard part
4. **Paraphrase stability** — cross-template τ=0.54 (5/6 pairs p<0.05); original-formal τ=0.857
5. **MCC-CE-v2** — error rates 23-40% (floor effect fixed), but 6/7 models 0% self-flagging
6. **Agent battery** — WMF-AM predicts 10-task deterministic battery: τ=0.612 (p=0.0003), partial τ|compl=0.411 (p=0.011)

---

## Environment

- **Server SSH:** `ssh -i ~/.ssh/windows_key hou@130.34.234.40` (alias: `sms-server`)
- **Project path on server:** `~/Research/CEF-NeurIPS2026/` (after sync)
- **Conda:** `~/miniconda3/bin/conda run -n py311 python`
- **Compiler:** `tectonic` (pdflatex broken — missing TeXLive Perl module)
- **GPU:** c03 A100 (4×) recommended; c04 RTX Pro 6000 (2×) also available
- **Scheduler:** SLURM (`sbatch`, `squeue`)

### Compile paper
```bash
cd ~/Research/CEF-NeurIPS2026/paper
tectonic cef_paper_v7.tex
```

### Regenerate all figures
```bash
~/miniconda3/bin/conda run -n py311 python figures/gen_figures_v8.py
```

---

## Immediate Next Steps (post Mar 18)

1. **arXiv submission** — polish abstract, check formatting, submit
2. **N expansion to 15+** — OpenRouter API (GPT-4o, Claude 3.5, Gemini Pro)
3. **Factor analysis** — CFA/EFA needs N≥15
4. **Standalone arithmetic control** — last-operation-only probe

---

## Skills Available

Skills are in `~/.claude/skills/`:
- `/auto-paper-improvement-loop` — iterative review (target 9/10 pre-submission)
- `/run-experiment` — deploy to server via SLURM
- `/monitor-experiment` — check job status
- `/analyze-results` — statistical analysis

Codex MCP: `mcp__codex__codex` + `mcp__codex__codex-reply` (GPT-5.4 xhigh reviewer)
