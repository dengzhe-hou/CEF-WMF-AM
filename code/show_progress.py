"""Quick script to show Phase 1 progress."""
import json
import numpy as np

d = json.load(open("/home/hou/Research/Cognitive-LLM-Agent-Position/project/data/results/cef_phase1_full.json"))
results = d["results"]
models = sorted(set(r["model"] for r in results))
dims = ["WMF-AM","WMF-IM","WMF-IR","MCC-MA","MCC-CE","EMC-TO","CLA-DC","CLA-RA","CLA-CR","VALIDITY-MMLU","VALIDITY-GSM8K"]

hdr = f"{'Model':<25} " + " ".join(f"{d:>8}" for d in dims)
print(hdr)
print("-" * len(hdr))

for m in models:
    vals = []
    for dim in dims:
        items = [r for r in results if r["model"] == m and r["sub_dim"] == dim]
        if not items:
            vals.append("       -")
        elif "accurate" in items[0]:
            acc = sum(r["accurate"] for r in items) / len(items)
            vals.append(f"{acc:>8.3f}")
        elif "ma_jaccard" in items[0]:
            v = np.mean([r["ma_jaccard"] for r in items])
            vals.append(f"{v:>8.3f}")
        elif "tau" in items[0]:
            v = np.mean([r["tau"] for r in items])
            vals.append(f"{v:>8.3f}")
        elif "recovery" in items[0]:
            v = np.mean([r["recovery"] for r in items])
            vals.append(f"{v:>8.3f}")
        elif "response_length" in items[0]:
            from scipy.stats import pearsonr
            diffs = [r["difficulty"] for r in items]
            lens = [r["response_length"] for r in items]
            if len(set(diffs)) > 1 and len(set(lens)) > 1:
                corr, _ = pearsonr(diffs, lens)
                vals.append(f"{corr:>8.3f}")
            else:
                vals.append("       -")
        elif "p_flag_given_wrong" in items[0]:
            vs = [r["p_flag_given_wrong"] for r in items if r["p_flag_given_wrong"] is not None]
            vals.append(f"{np.mean(vs):>8.3f}" if vs else "       -")
        else:
            vals.append("       ?")
    print(f"{m:<25} " + " ".join(vals))

print(f"\nModels completed: {len(models)}/7")
print(f"Total trials: {len(results)}")
