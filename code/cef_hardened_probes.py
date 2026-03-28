"""
CEF Hardened Probes — Fix ceiling effects in WMF-IM, WMF-IR, CLA-CR

Current ceiling effects (Phase 1 data):
  WMF-IM: 0.975 (all models near-perfect)
  WMF-IR: 0.983-1.000 (no differentiation)
  CLA-CR: 1.000 (trivial recovery)

Fixes:
  WMF-IM-hard: 10-20 targets, deep semantic interference, multi-hop queries
  WMF-IR-hard: 3+ overlapping lists, similar entity names, delayed query
  CLA-CR-hard: Medium baseline, brutal overload block, 5-block paradigm

Usage:
    python cef_hardened_probes.py --phase ollama --seeds 2
    python cef_hardened_probes.py --models ollama:qwen2.5:7b ollama:llama3.1:8b
"""

import argparse
import json
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, RESULTS_DIR, call_model

OLLAMA_MODELS = [k for k in MODELS if MODELS[k]["provider"] == "ollama"]
API_MODELS = [k for k in MODELS if MODELS[k]["provider"] != "ollama"]

API_DELAY = 1.0
OLLAMA_DELAY = 0.3


def _delay(model):
    cfg = MODELS.get(model, {})
    if cfg.get("provider") == "ollama":
        time.sleep(OLLAMA_DELAY)
    else:
        time.sleep(API_DELAY)


# ── WMF-IM-hard: High-load Information Maintenance ─────────────────────────

# Expanded entity bank with similar names (interference source)
IM_ENTITIES = [
    ("Alice", "collected", "stamps"), ("Alicia", "collected", "coins"),
    ("Bob", "scored", "points"), ("Bobby", "scored", "goals"),
    ("Carol", "planted", "trees"), ("Caroline", "planted", "flowers"),
    ("David", "wrote", "letters"), ("Daniel", "wrote", "poems"),
    ("Emma", "sold", "tickets"), ("Emily", "sold", "books"),
    ("Frank", "caught", "fish"), ("Frederick", "caught", "butterflies"),
    ("Grace", "baked", "pies"), ("Gloria", "baked", "cakes"),
    ("Henry", "built", "models"), ("Harold", "built", "shelves"),
    ("Iris", "painted", "portraits"), ("Ivy", "painted", "landscapes"),
    ("James", "repaired", "clocks"), ("Jason", "repaired", "watches"),
]

# Similar-domain distractors (harder interference)
IM_DISTRACTORS = [
    "Meanwhile, a survey reported that the average person {verb} approximately {n} {obj} per year.",
    "In a related study, participants who {verb} more than {n} {obj} showed higher satisfaction.",
    "Historical records indicate that in 1850, a typical household {verb} about {n} {obj}.",
    "A recent census found that {n} percent of respondents regularly {verb} {obj}.",
]


def build_wmf_im_hard(rng, n_targets, interference_depth="deep"):
    """
    Harder WMF-IM: many targets, semantically similar distractors, multi-hop query.
    """
    selected = rng.sample(IM_ENTITIES, min(n_targets, len(IM_ENTITIES)))
    targets = []
    for entity, verb, obj in selected:
        val = rng.randint(10, 99)
        targets.append({
            "entity": entity, "verb": verb, "obj": obj, "value": val,
            "sentence": f"{entity} {verb} {val} {obj}."
        })

    # Deep interference: semantically similar distractors
    distractor_sentences = []
    for t in targets[:n_targets // 2]:
        template = rng.choice(IM_DISTRACTORS)
        fake_val = rng.randint(10, 99)
        while fake_val == t["value"]:
            fake_val = rng.randint(10, 99)
        distractor_sentences.append(
            template.format(verb=t["verb"], n=fake_val, obj=t["obj"])
        )

    # Interleave
    all_sentences = [t["sentence"] for t in targets] + distractor_sentences
    rng.shuffle(all_sentences)

    # Multi-hop query: ask about difference or sum
    if len(targets) >= 2 and rng.random() < 0.5:
        # Multi-hop: "How many more X did A collect than B?"
        t1, t2 = rng.sample(targets, 2)
        diff = abs(t1["value"] - t2["value"])
        if t1["value"] >= t2["value"]:
            query = (f"How many more {t1['obj']} did {t1['entity']} {t1['verb']} "
                     f"than {t2['entity']}? Reply with ONLY the number.")
        else:
            query = (f"How many more {t2['obj']} did {t2['entity']} {t2['verb']} "
                     f"than {t1['entity']}? Reply with ONLY the number.")
        correct = diff
        query_type = "multi_hop_diff"
    else:
        # Standard single-entity query
        query_target = rng.choice(targets)
        query = (f"How many {query_target['obj']} did {query_target['entity']} "
                 f"{query_target['verb']}? Reply with ONLY the number.")
        correct = query_target["value"]
        query_type = "single"

    prompt = f"""Read the following passage carefully, then answer the question.

---
{' '.join(all_sentences)}
---

{query}"""

    return prompt, correct, query_type


def run_wmf_im_hard(model, seeds):
    """WMF-IM-hard: load levels 5, 8, 12, 16, 20."""
    results = []
    load_levels = [5, 8, 12, 16, 20]
    probes_per_level = 5

    for seed in seeds:
        rng = random.Random(seed)
        for n in load_levels:
            for _ in range(probes_per_level):
                prompt, correct, qtype = build_wmf_im_hard(rng, n)
                try:
                    resp = call_model(model, prompt)
                    nums = re.findall(r"-?\d+", resp)
                    pred = int(nums[0]) if nums else -1
                except Exception:
                    pred = -999
                results.append({
                    "probe": "WMF-IM-hard", "model": model, "seed": seed,
                    "n_targets": n, "query_type": qtype,
                    "correct": correct, "predicted": pred,
                    "accurate": int(pred == correct),
                })
                _delay(model)
    return results


# ── WMF-IR-hard: Multi-list Interference Resistance ────────────────────────

def build_wmf_ir_hard(rng, n_items, n_lists=3):
    """
    Harder WMF-IR: 3+ overlapping lists with similar entity names.
    Query about a specific list after all lists presented.
    """
    # Use similar-name pairs to increase confusion
    all_pairs = list(zip(IM_ENTITIES[::2], IM_ENTITIES[1::2]))  # paired by similarity
    selected_pairs = rng.sample(all_pairs, min(n_items, len(all_pairs)))

    lists = []
    for list_idx in range(n_lists):
        items = []
        for pair in selected_pairs:
            # Alternate which name from each pair goes in which list
            entity_tuple = pair[list_idx % 2] if list_idx < 2 else pair[rng.randint(0, 1)]
            entity, verb, obj = entity_tuple
            val = rng.randint(10, 99)
            items.append({
                "entity": entity, "verb": verb, "obj": obj, "value": val,
                "sentence": f"{entity} {verb} {val} {obj}."
            })
        lists.append(items)

    # Present all lists
    list_texts = []
    for i, lst in enumerate(lists):
        label = ["First", "Second", "Third", "Fourth", "Fifth"][i]
        text = f"{label} report:\n" + " ".join(item["sentence"] for item in lst)
        list_texts.append(text)

    # Query about the first list
    query_item = rng.choice(lists[0])
    prompt = f"""You will read {n_lists} separate reports about similar activities. After reading ALL reports, answer a question about the FIRST report ONLY.

{chr(10).join(list_texts)}

Question: According to the FIRST report only, how many {query_item['obj']} did {query_item['entity']} {query_item['verb']}?

Reply with ONLY the number."""

    return prompt, query_item["value"]


def run_wmf_ir_hard(model, seeds):
    """WMF-IR-hard: 3-4 lists, 3-7 items per list."""
    results = []
    configs = [(3, 3), (3, 5), (4, 3), (4, 5), (3, 7)]
    probes_per_config = 4

    for seed in seeds:
        rng = random.Random(seed)
        for n_lists, n_items in configs:
            for _ in range(probes_per_config):
                prompt, correct = build_wmf_ir_hard(rng, n_items, n_lists)
                try:
                    resp = call_model(model, prompt)
                    nums = re.findall(r"-?\d+", resp)
                    pred = int(nums[0]) if nums else -1
                except Exception:
                    pred = -999
                results.append({
                    "probe": "WMF-IR-hard", "model": model, "seed": seed,
                    "n_lists": n_lists, "n_items": n_items,
                    "correct": correct, "predicted": pred,
                    "accurate": int(pred == correct),
                })
                _delay(model)
    return results


# ── CLA-CR-hard: Brutal Cognitive Recovery ─────────────────────────────────

# Medium-difficulty baseline (not trivially easy)
CLA_MEDIUM = [
    {"q": "What is 23 × 17?", "a": "391"},
    {"q": "What is the capital of Slovakia?", "a": "Bratislava"},
    {"q": "What is 144 ÷ 12?", "a": "12"},
    {"q": "Who wrote 'The Great Gatsby'?", "a": "F. Scott Fitzgerald"},
    {"q": "What is the square root of 256?", "a": "16"},
    {"q": "What element has the symbol Ag?", "a": "Silver"},
    {"q": "What is 15% of 240?", "a": "36"},
    {"q": "In which year was the UN founded?", "a": "1945"},
    {"q": "What is 7^3?", "a": "343"},
    {"q": "What is the chemical formula for table salt?", "a": "NaCl"},
]

# Brutal overload block
CLA_BRUTAL = [
    {"q": "What is 347 × 283?", "a": "98201"},
    {"q": "What is the 25th prime number?", "a": "97"},
    {"q": "What is the sum of all prime numbers between 50 and 100?", "a": "732"},
    {"q": "What is 19^3 - 17^3?", "a": "1946"},
    {"q": "If a right triangle has legs 7 and 24, what is the hypotenuse?", "a": "25"},
    {"q": "What is the determinant of [[5,3,1],[2,4,6],[1,7,2]]?", "a": "-98"},
    {"q": "Who was the 14th President of the United States?", "a": "Franklin Pierce"},
    {"q": "What is 2^17?", "a": "131072"},
    {"q": "In which year was the Peace of Westphalia signed?", "a": "1648"},
    {"q": "What is the integral of x*ln(x)?", "a": "x^2*ln(x)/2 - x^2/4 + C"},
    {"q": "What is 983 × 764?", "a": "751012"},
    {"q": "What was the population of London in 1800 (approximate, nearest 100k)?", "a": "1000000"},
    {"q": "What is the 30th element of the Fibonacci sequence?", "a": "832040"},
    {"q": "What is the GCD of 1071 and 462?", "a": "21"},
    {"q": "Who was the last Merovingian king?", "a": "Childeric III"},
]


def run_cla_cr_hard(model, seeds):
    """
    5-block paradigm: Medium1 → Brutal1 → Medium2 → Brutal2 → Medium3
    Recovery = accuracy(Medium3) / accuracy(Medium1)
    Also measure: cumulative fatigue = accuracy(Medium2) vs Medium1
    """
    results = []

    for seed in seeds:
        rng = random.Random(seed)

        # Create 3 medium blocks and 2 brutal blocks
        m1 = list(CLA_MEDIUM)
        m2 = list(CLA_MEDIUM)
        m3 = list(CLA_MEDIUM)
        b1 = list(CLA_BRUTAL)
        b2 = list(CLA_BRUTAL)
        rng.shuffle(m1)
        rng.shuffle(m2)
        rng.shuffle(m3)
        rng.shuffle(b1)
        rng.shuffle(b2)

        blocks = [
            ("M1", m1), ("B1", b1), ("M2", m2), ("B2", b2), ("M3", m3)
        ]

        block_results = {}
        history = []

        for block_name, problems in blocks:
            q_block = "\n".join(
                f"Q{i+1}: {p['q']}" for i, p in enumerate(problems)
            )
            prompt = f"""Answer each question. Give ONLY the answer, one per line.

{q_block}

Format: "Q1: answer"."""

            try:
                resp = call_model(model, prompt, history=history)
                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": resp})
            except Exception:
                resp = ""
            _delay(model)

            # Parse and score
            answers = {}
            for line in resp.split("\n"):
                m_match = re.match(r"Q(\d+):\s*(.+)", line.strip())
                if m_match:
                    answers[int(m_match.group(1))] = m_match.group(2).strip()

            correct_count = 0
            total = len(problems)
            for i, p in enumerate(problems):
                ma = answers.get(i + 1, "")
                correct_ans = p["a"].lower().replace(",", "").replace(" ", "")
                model_ans = ma.lower().replace(",", "").replace(" ", "").strip()
                if correct_ans in model_ans or model_ans in correct_ans:
                    correct_count += 1

            block_results[block_name] = correct_count / total if total > 0 else 0.0

        # Compute metrics
        m1_acc = block_results.get("M1", 0.0)
        m2_acc = block_results.get("M2", 0.0)
        m3_acc = block_results.get("M3", 0.0)
        b1_acc = block_results.get("B1", 0.0)
        b2_acc = block_results.get("B2", 0.0)

        recovery = m3_acc / m1_acc if m1_acc > 0 else 0.0
        mid_fatigue = m2_acc / m1_acc if m1_acc > 0 else 0.0
        overload_perf = (b1_acc + b2_acc) / 2

        results.append({
            "probe": "CLA-CR-hard", "model": model, "seed": seed,
            "m1_acc": round(m1_acc, 4),
            "b1_acc": round(b1_acc, 4),
            "m2_acc": round(m2_acc, 4),
            "b2_acc": round(b2_acc, 4),
            "m3_acc": round(m3_acc, 4),
            "recovery": round(recovery, 4),
            "mid_fatigue": round(mid_fatigue, 4),
            "overload_perf": round(overload_perf, 4),
        })

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def run_model(model, seeds):
    """Run all hardened probes for one model."""
    all_results = []
    t0 = time.time()

    print(f"\n{'=' * 60}")
    print(f"  Model: {model}")
    print(f"  Seeds: {seeds}")
    print(f"  Probes: WMF-IM-hard, WMF-IR-hard, CLA-CR-hard")
    print(f"  Start: {datetime.now().isoformat()}")
    print(f"{'=' * 60}")

    # WMF-IM-hard
    print(f"  [1/3] WMF-IM-hard (5 loads × 5 probes × {len(seeds)} seeds)...")
    try:
        r = run_wmf_im_hard(model, seeds)
        acc = sum(x["accurate"] for x in r) / len(r) if r else 0
        # By load level
        by_n = {}
        for x in r:
            by_n.setdefault(x["n_targets"], []).append(x["accurate"])
        curve = {n: round(np.mean(v), 3) for n, v in sorted(by_n.items())}
        print(f"         → {len(r)} trials, mean={acc:.3f}, curve: {curve}")
        all_results.extend(r)
    except Exception as e:
        print(f"         → ERROR: {e}")
        traceback.print_exc()

    # WMF-IR-hard
    print(f"  [2/3] WMF-IR-hard (5 configs × 4 probes × {len(seeds)} seeds)...")
    try:
        r = run_wmf_ir_hard(model, seeds)
        acc = sum(x["accurate"] for x in r) / len(r) if r else 0
        print(f"         → {len(r)} trials, mean={acc:.3f}")
        all_results.extend(r)
    except Exception as e:
        print(f"         → ERROR: {e}")
        traceback.print_exc()

    # CLA-CR-hard
    print(f"  [3/3] CLA-CR-hard (5-block paradigm × {len(seeds)} seeds)...")
    try:
        r = run_cla_cr_hard(model, seeds)
        if r:
            avg_rec = np.mean([x["recovery"] for x in r])
            avg_fat = np.mean([x["mid_fatigue"] for x in r])
            print(f"         → {len(r)} rounds, recovery={avg_rec:.3f}, mid_fatigue={avg_fat:.3f}")
        all_results.extend(r)
    except Exception as e:
        print(f"         → ERROR: {e}")
        traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n  Total: {len(all_results)} trials in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="CEF Hardened Probes")
    parser.add_argument("--phase", choices=["ollama", "api", "all"], default="ollama")
    parser.add_argument("--models", nargs="+", help="Specific models")
    parser.add_argument("--seeds", type=int, default=2, help="Number of seeds")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))

    if args.models:
        models = args.models
    elif args.phase == "ollama":
        models = OLLAMA_MODELS
    elif args.phase == "api":
        models = API_MODELS
    else:
        models = list(MODELS.keys())

    # Exclude llama3.1:70b (timeout issues)
    models = [m for m in models if "70b" not in m]

    print(f"CEF Hardened Probes")
    print(f"Models: {len(models)}")
    print(f"Seeds: {seeds}")
    print(f"Start: {datetime.now().isoformat()}")

    all_results = []
    for model in models:
        try:
            r = run_model(model, seeds)
            all_results.extend(r)
        except Exception as e:
            print(f"\nFATAL ERROR for {model}: {e}")
            traceback.print_exc()

        # Incremental save
        out_name = args.output or f"cef_hardened_{args.phase}.json"
        out_path = RESULTS_DIR / out_name
        with open(out_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "seeds": seeds,
                "models_completed": list(set(r["model"] for r in all_results)),
                "total_trials": len(all_results),
                "results": all_results,
            }, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("HARDENED PROBES SUMMARY")
    print(f"{'=' * 70}")
    probes = sorted(set(r["probe"] for r in all_results))
    for model in models:
        print(f"\n{model}:")
        for p in probes:
            items = [r for r in all_results if r["model"] == model and r["probe"] == p]
            if not items:
                continue
            if "accurate" in items[0]:
                acc = sum(r["accurate"] for r in items) / len(items)
                print(f"  {p}: {acc:.3f} (n={len(items)})")
            elif "recovery" in items[0]:
                rec = np.mean([r["recovery"] for r in items])
                fat = np.mean([r["mid_fatigue"] for r in items])
                print(f"  {p}: recovery={rec:.3f}, fatigue={fat:.3f} (n={len(items)})")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
