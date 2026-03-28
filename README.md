# CEF-WMF-AM

Code and data for:

**Beyond Completion: Probing Cumulative State Tracking to Predict LLM Agent Performance**

Dengzhe Hou, Lingyu Jiang, Deng Li, Zirui Li, Fangzhou Lin, Kazunori D Yamada

*arXiv preprint, 2026*

## Key Result

WMF-AM, a calibrated no-scratchpad probe of cumulative arithmetic state tracking, predicts downstream agent performance across 20 open-weight models (0.5B–35B, 13 families) with Kendall's τ = 0.612 (p < 0.001, pre-specified, Bonferroni-corrected).

## Repository Structure

```
code/               Experiment scripts (WMF-AM probes, agent battery, ablations)
data/               Per-model JSON result files
PREREGISTRATION.md  Pre-registered analysis plan
```

## Quick Start

```bash
# Run WMF-AM probe on a model
conda run -n py311 python code/wmf_am_multiseed_expansion.py
```

## License

MIT
