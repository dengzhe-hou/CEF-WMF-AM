"""Show progress for all CEF experiments."""
import json
import os
import numpy as np

RESULTS = "/home/hou/Research/Cognitive-LLM-Agent-Position/project/data/results"

# 1. Convergent validity
path = os.path.join(RESULTS, "cef_convergent_validity_ollama.json")
if os.path.exists(path):
    d = json.load(open(path))
    r = d.get("results", [])
    models = sorted(set(x["model"] for x in r))
    print(f"=== CONVERGENT VALIDITY ({len(models)}/7 models) ===")
    for m in models:
        rfp = np.mean([x["process_score"] for x in r if x["model"] == m and x.get("probe") == "RF-POC"])
        sk = np.mean([x["accurate"] for x in r if x["model"] == m and x.get("probe") == "Self-Knowledge"])
        fr = np.mean([x["accurate"] for x in r if x["model"] == m and x.get("probe") == "Factual-Retrieval"])
        print(f"  {m}: RF-POC={rfp:.3f}  Self-Know={sk:.3f}  Factual={fr:.3f}")

# 2. Hardened probes
print()
path2 = os.path.join(RESULTS, "cef_hardened_results.json")
if os.path.exists(path2):
    d2 = json.load(open(path2))
    r2 = d2.get("results", [])
    m2 = sorted(set(x["model"] for x in r2))
    print(f"=== HARDENED PROBES ({len(m2)}/7 models, {len(r2)} trials) ===")
    for m in m2:
        mr = [x for x in r2 if x["model"] == m]
        acc_items = [x for x in mr if "accurate" in x]
        rec_items = [x for x in mr if "recovery" in x]
        ovl_items = [x for x in mr if "overload_perf" in x]
        parts = []
        if acc_items:
            parts.append("WMF-hard={:.3f}".format(np.mean([x["accurate"] for x in acc_items])))
        if ovl_items:
            parts.append("overload={:.3f}".format(np.mean([x["overload_perf"] for x in ovl_items])))
        if rec_items:
            parts.append("recovery={:.3f}".format(np.mean([x["recovery"] for x in rec_items])))
        print("  {}: {}".format(m, "  ".join(parts)))

# 3. MCC-CE v2
print()
path3 = os.path.join(RESULTS, "cef_mcc_ce_v2_ollama.json")
if os.path.exists(path3):
    d3 = json.load(open(path3))
    r3 = d3.get("results", [])
    m3 = sorted(set(x["model"] for x in r3))
    print("=== MCC-CE V2 ({}/7 models, {} trials) ===".format(len(m3), len(r3)))
    for m in m3:
        mr = [x for x in r3 if x["model"] == m]
        keys = set()
        for x in mr:
            for k, v in x.items():
                if isinstance(v, (int, float)) and k not in ("seed", "trial"):
                    keys.add(k)
        parts = []
        for k in sorted(keys):
            vs = [x[k] for x in mr if k in x and x[k] is not None and isinstance(x[k], (int, float))]
            if vs:
                parts.append("{}={:.3f}".format(k, np.mean(vs)))
        print("  {}: {}".format(m, ", ".join(parts[:6])))
else:
    print("MCC-CE v2: not started yet")

# 4. Agent validation
print()
path4 = os.path.join(RESULTS, "cef_agent_validation_ollama.json")
if os.path.exists(path4):
    d4 = json.load(open(path4))
    r4 = d4.get("results", [])
    m4 = sorted(set(x["model"] for x in r4))
    print("=== AGENT VALIDATION ({}/7 models, {} trials) ===".format(len(m4), len(r4)))
else:
    print("Agent validation: not started yet")
