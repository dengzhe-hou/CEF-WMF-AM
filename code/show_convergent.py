"""Show convergent validity results."""
import json
import numpy as np

d = json.load(open("/home/hou/Research/Cognitive-LLM-Agent-Position/project/data/results/cef_convergent_validity_ollama.json"))
print("Models done:", d.get("models_completed", []))
print("Total trials:", d.get("total_trials", 0))
print()

results = d["results"]
models = sorted(set(r["model"] for r in results))
probes = sorted(set(r["sub_dim"] for r in results))

# Header
hdr = "Model".ljust(25) + "  ".join(p.rjust(18) for p in probes)
print(hdr)
print("-" * len(hdr))

for m in models:
    vals = []
    for p in probes:
        items = [r for r in results if r["model"] == m and r["sub_dim"] == p]
        if not items:
            vals.append("     -")
            continue
        # Try different score keys
        for key in ["process_score", "score", "accurate", "accuracy"]:
            scores = [r[key] for r in items if key in r]
            if scores:
                vals.append(f"{np.mean(scores):>18.3f}")
                break
        else:
            vals.append("     ?")
    print(m.ljust(25) + "  ".join(vals))
