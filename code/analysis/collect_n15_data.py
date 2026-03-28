"""Collect and summarize all N=15 CEF data."""
import json
import glob

RESULTS = "/home/hou/Research/Cognitive-LLM-Agent-Position/project/data/results"

# Read all N-expansion results
print("=== Expansion 8 models ===")
for f in sorted(glob.glob(f"{RESULTS}/cef_nexp_ollama_*[0-9].json")):
    with open(f) as fp:
        d = json.load(fp)
    m = d.get("model", "?")
    oc = d.get("outcome_correctness", {})
    wm = d.get("wmf_am", {})
    yk = d.get("yoked_control", {})
    mc = d.get("mcc_ma", {})
    oc_s = oc.get("score", "--")
    wm_m = wm.get("mean", "--")
    yk_m = yk.get("mean", "--")
    mc_r = mc.get("monitoring_r", "--")
    print(f"  {m}: outcome={oc_s} wmf_am={wm_m} yoked={yk_m} mcc_r={mc_r}")
    if wm.get("by_depth"):
        print(f"    wmf_am_by_depth: {wm['by_depth']}")
    if yk.get("by_depth"):
        print(f"    yoked_by_depth: {yk['by_depth']}")

# Read original 7 data
print()
print("=== Original 7 models ===")
with open(f"{RESULTS}/cef_expanded.json") as fp:
    exp = json.load(fp)
for m in exp.get("per_model", []):
    name = m["model"]
    comp = m.get("completion", {}).get("accuracy", "--")
    wmf = m.get("wmf_am", {}).get("mean_accuracy", "--")
    print(f"  {name}: completion={comp} wmf_am={wmf}")

# Read multiseed data
print()
print("=== Multiseed WMF-AM (original 7) ===")
with open(f"{RESULTS}/cef_wmf_multiseed.json") as fp:
    ms = json.load(fp)
for m in ms.get("per_model", []):
    name = m["model"]
    mean_a = m.get("mean_accuracy", "--")
    sd_a = m.get("sd_accuracy", "--")
    by_k = m.get("by_depth", {})
    print(f"  {name}: mean={mean_a} sd={sd_a} by_k={by_k}")

# Read yoked control data
print()
print("=== Yoked control (original 7) ===")
for f in sorted(glob.glob(f"{RESULTS}/wmf_am_yoked_control_ollama_*.json")):
    with open(f) as fp:
        d = json.load(fp)
    s = d.get("summary", {})
    m = d.get("model", "?")
    ov = s.get("overall_accuracy", "--")
    print(f"  {m}: yoked_overall={ov}")
