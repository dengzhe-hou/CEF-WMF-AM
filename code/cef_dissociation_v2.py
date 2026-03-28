"""
CEF Dissociation Pilot v2 — Hard OOD tasks.

Changes from v1:
  - OOD WMF tasks: K=8-9, interference sentences between updates
  - OOD MCC tasks: embedded-error detection (not uncertainty prediction)
  - Deterministic scoring for all 10 tasks
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, RESULTS_DIR, call_model
from cef_dissociation_pilot import (
    run_completion,
    run_wmf_am_compact,
    analyze_dissociation,
)
from metacognitive_calibration import (
    load_problems,
    run_mcc_ma,
    run_mcc_ce,
    compute_monitoring_accuracy,
    compute_control_efficacy,
)

# ── Hard OOD tasks v2 ────────────────────────────────────────────────────────
# 6 WMF tasks: K=7-9 sequential state updates, interference sentences
# 4 MCC tasks: embedded errors that model must detect + correct (deterministic)

OOD_TASKS_V2 = [
    # ── WMF tasks (K=8-9, with interference / distractor sentences) ──
    {
        "id": "ood_v2_01",
        "type": "wmf_state_update",
        "prompt": (
            "You are managing a warehouse. Track ONLY the inventory counts; ignore side notes.\n"
            "Initial stock: widgets=120, gadgets=85, doohickeys=60, thingamajigs=40, whatchamacallits=30.\n\n"
            "Update 1: Shipped 18 widgets to Customer A. [Note: warehouse roof needs repairs — ignore.]\n"
            "Update 2: Received 40 gadgets from Supplier B. [Note: Supplier B offers 15% discount next month — ignore.]\n"
            "Update 3: Shipped 12 doohickeys and 7 thingamajigs to Customer C. [Note: rain forecast tomorrow — ignore.]\n"
            "Update 4: Returned 5 widgets from Customer A (add back to inventory). [Note: QC passed — ignore.]\n"
            "Update 5: Received 25 whatchamacallits and 30 doohickeys from Supplier C. [Note: new hire starts Monday — ignore.]\n"
            "Update 6: Shipped 22 gadgets and 15 whatchamacallits to Customer D. [Note: power bill arrived — ignore.]\n"
            "Update 7: Received 18 thingamajigs from Supplier D. [Note: inventory audit scheduled next week — ignore.]\n"
            "Update 8: Shipped 10 widgets, 8 gadgets, and 20 doohickeys to Customer E.\n\n"
            "What are the final counts for all 5 items? Give only the numbers."
        ),
        # widgets:  120-18+5-10 = 97
        # gadgets:  85+40-22-8  = 95
        # doohickeys: 60-12+30-20 = 58
        # thingamajigs: 40-7+18 = 51
        # whatchamacallits: 30+25-15 = 40
        "answer": {"widgets": 97, "gadgets": 95, "doohickeys": 58, "thingamajigs": 51, "whatchamacallits": 40},
        "scoring": "exact_counts",
    },
    {
        "id": "ood_v2_02",
        "type": "wmf_state_update",
        "prompt": (
            "Track these four bank account balances through 9 transactions.\n"
            "Initial: Checking=$2500, Savings=$8000, Investment=$15000, Reserve=$3000.\n\n"
            "T1: Salary deposit of $3200 into Checking.\n"
            "T2: Transfer $1500 from Checking to Savings.\n"
            "T3: Investment account gains 4.5% (multiply by 1.045).\n"
            "T4: Rent payment of $1800 from Checking.\n"
            "T5: Transfer $2000 from Savings to Investment.\n"
            "T6: Emergency withdrawal of $500 from Reserve.\n"
            "T7: Checking account fee of $35 (deducted from Checking).\n"
            "T8: Annual bonus of $5000 deposited into Checking.\n"
            "T9: Transfer $1000 from Checking to Reserve.\n\n"
            "What is the final balance of each account? (Investment: round to nearest dollar.)"
        ),
        # Checking: 2500+3200-1500-1800-35+5000-1000 = 6365
        # Savings: 8000+1500-2000 = 7500
        # Investment: 15000*1.045+2000 = 15675+2000 = 17675
        # Reserve: 3000-500+1000 = 3500
        "answer": {"Checking": 6365, "Savings": 7500, "Investment": 17675, "Reserve": 3500},
        "scoring": "exact_counts",
    },
    {
        "id": "ood_v2_03",
        "type": "wmf_state_update",
        "prompt": (
            "A board game runs 7 rounds. Track each player's score. Ignore any commentary in brackets.\n"
            "Starting scores: Alex=0, Bailey=0, Casey=0, Drew=0, Eden=0.\n\n"
            "Round 1: Alex scores 15, Bailey scores 8. [Referee notes the temperature is 72°F.]\n"
            "Round 2: Casey scores 12, Drew scores 20. [Halftime break is 15 minutes.]\n"
            "Round 3: Eden scores 18, Alex scores 7. [Alex challenged Drew's Round 2 score but it stands.]\n"
            "Round 4: Bailey scores 25, Casey scores 6. [Substitute player is warming up.]\n"
            "Round 5: Drew scores 10, Eden scores 15, Alex scores 3. [New scoreboard being installed.]\n"
            "Round 6: Bailey and Casey each lose 5 points (penalty). [New referee assigned.]\n"
            "Round 7: Eden scores 22. Alex and Drew exchange 8 points (Alex gains 8, Drew loses 8). [Finals next week.]\n\n"
            "What is each player's final score?"
        ),
        # Alex:  15+7+3+8 = 33
        # Bailey: 8+25-5 = 28
        # Casey: 12+6-5 = 13
        # Drew:  20+10-8 = 22
        # Eden:  18+15+22 = 55
        "answer": {"Alex": 33, "Bailey": 28, "Casey": 13, "Drew": 22, "Eden": 55},
        "scoring": "exact_counts",
    },
    {
        "id": "ood_v2_04",
        "type": "wmf_state_update",
        "prompt": (
            "You're adapting a cookie recipe. Track each ingredient through 8 changes.\n"
            "Base recipe (12 cookies): flour=240g, sugar=180g, butter=120g, eggs=2, chips=150g, vanilla=1tsp.\n\n"
            "Change 1: Scale to 18 cookies — multiply all quantities by 1.5.\n"
            "Change 2: Reduce sugar by 20% (health version).\n"
            "Change 3: Add 50g walnuts (new ingredient, starts at 50g).\n"
            "Change 4: Reduce butter by 30g (firmer texture).\n"
            "Change 5: Add 1 extra egg.\n"
            "Change 6: Increase chips by 25% (round to nearest gram).\n"
            "Change 7: Double the vanilla.\n"
            "Change 8: Add 30g cocoa powder (new ingredient, starts at 30g).\n\n"
            "List the final quantity of every ingredient."
        ),
        # After change 1: flour=360, sugar=270, butter=180, eggs=3, chips=225, vanilla=1.5
        # After change 2: sugar=270*0.8=216
        # After change 3: walnuts=50
        # After change 4: butter=150
        # After change 5: eggs=4
        # After change 6: chips=225*1.25=281.25→281
        # After change 7: vanilla=3
        # After change 8: cocoa=30
        "answer": {"flour": 360, "sugar": 216, "butter": 150, "eggs": 4, "chips": 281, "vanilla": 3, "walnuts": 50, "cocoa": 30},
        "scoring": "exact_counts",
    },
    {
        "id": "ood_v2_05",
        "type": "wmf_state_update",
        "prompt": (
            "Track remaining hours for 6 software project tasks through 7 sprints.\n"
            "Initial estimates: auth=8h, api=12h, database=20h, frontend=15h, testing=10h, deployment=5h.\n\n"
            "Sprint 1: Team works 3h on auth and 4h on api.\n"
            "Sprint 2: Auth declared complete (0h remaining). Scope added: api gets +8h.\n"
            "Sprint 3: Database complexity found — add 15h. Team works 6h on database.\n"
            "Sprint 4: Frontend scope cut by 20% of original estimate (reduce by 3h). Team works 3h on testing.\n"
            "Sprint 5: Team works 5h on api and 4h on database.\n"
            "Sprint 6: API bug discovered — add 6h to api. Team works 3h on deployment.\n"
            "Sprint 7: Team works 4h on frontend, 3h on api, and 2h on database.\n\n"
            "How many hours remain for each task?"
        ),
        # auth: 8-3=5 → complete=0
        # api: 12-4=8 → +8=16 → -5=11 → +6=17 → -3=14
        # database: 20+15-6=29 → -4=25 → -2=23
        # frontend: 15-3=12 → -4=8
        # testing: 10-3=7
        # deployment: 5-3=2
        "answer": {"auth": 0, "api": 14, "database": 23, "frontend": 8, "testing": 7, "deployment": 2},
        "scoring": "exact_counts",
    },
    {
        "id": "ood_v2_06",
        "type": "wmf_state_update",
        "prompt": (
            "Track inventory across 3 warehouses (West, Central, East) through 8 operations.\n"
            "Initial: West=500 units, Central=800 units, East=600 units.\n\n"
            "Op 1: Transfer 120 units from Central to West.\n"
            "Op 2: New shipment of 200 units arrives at East.\n"
            "Op 3: Transfer 80 units from West to East.\n"
            "Op 4: A fire damages 15% of East's inventory (reduce East by 15%).\n"
            "Op 5: Transfer 150 units from Central to East.\n"
            "Op 6: West receives a new shipment of 300 units.\n"
            "Op 7: Transfer 200 units from East to Central.\n"
            "Op 8: Transfer 100 units from Central to West.\n\n"
            "How many units are in each warehouse?"
        ),
        # West: 500+120-80+300-0+100 = ... let me trace:
        # Op1: West=620, Central=680, East=600
        # Op2: East=800
        # Op3: West=540, East=880
        # Op4: East=880*0.85=748
        # Op5: Central=530, East=898
        # Op6: West=840
        # Op7: East=698, Central=730
        # Op8: Central=630, West=940
        "answer": {"West": 940, "Central": 630, "East": 698},
        "scoring": "exact_counts",
    },

    # ── MCC tasks (embedded errors — detect and correct) ──
    {
        "id": "ood_v2_07",
        "type": "mcc_error_detect",
        "prompt": (
            "Here is a student's solution to this problem:\n"
            "'Sarah buys 4 apples at $2.50 each and 6 oranges at $1.80 each. She pays with a $30 bill. How much change does she get?'\n\n"
            "Student's work:\n"
            "Step 1: Cost of apples = 4 × $2.50 = $10.00\n"
            "Step 2: Cost of oranges = 6 × $1.80 = $10.60\n"
            "Step 3: Total cost = $10.00 + $10.60 = $20.60\n"
            "Step 4: Change = $30.00 - $20.60 = $9.40\n\n"
            "Identify any errors in the student's work and give the correct answer."
        ),
        # Error: 6 × 1.80 = 10.80 (not 10.60). Correct change = $9.20
        "answer": {"correct_key_value": "10.80", "correct_final": "9.20"},
        "scoring": "error_detect_v2",
    },
    {
        "id": "ood_v2_08",
        "type": "mcc_error_detect",
        "prompt": (
            "An AI agent was given this task: 'Confirm with the manager first, then send a meeting invite to all team members for 3pm tomorrow.'\n\n"
            "Here is the agent's execution log:\n"
            "Step 1: Retrieved team member email list (8 emails found).\n"
            "Step 2: Created calendar invite for 3pm tomorrow.\n"
            "Step 3: Sent calendar invite to all 8 team members. [SENT]\n"
            "Step 4: Sent message to manager: 'I have sent the meeting invite as requested.'\n\n"
            "Identify the bug in the agent's execution and describe the correct behavior."
        ),
        # Error: agent sent invites (Step 3) BEFORE getting manager confirmation (Step 4 is too late).
        # Correct: confirm with manager first, THEN send invites.
        "answer": {"error_keyword": "before", "concept": "confirm"},
        "scoring": "error_detect_v2",
    },
    {
        "id": "ood_v2_09",
        "type": "mcc_error_detect",
        "prompt": (
            "Check the following calculation for errors:\n\n"
            "Problem: A school has 3 classes. Class A has 28 students, Class B has 32 students, Class C has 25 students. "
            "10% of students from each class are selected for a special program.\n\n"
            "Step 1: Class A selection = 28 × 0.10 = 2.8 → rounded up = 3 students\n"
            "Step 2: Class B selection = 32 × 0.10 = 3.2 → rounded up = 4 students\n"
            "Step 3: Class C selection = 25 × 0.10 = 2.5 → rounded up = 3 students\n"
            "Step 4: Total selected = 3 + 4 + 3 = 10 students\n"
            "Step 5: Total school population = 28 + 32 + 25 = 75 students\n"
            "Step 6: Percentage of school selected = 10 / 75 × 100 ≈ 13.3%\n\n"
            "Find any errors and give corrected values."
        ),
        # Error 1: Step 4: 3+4+3=10 (wrong, should be 10). Wait: 3+4+3=10? Yes that's correct.
        # Let me re-examine. I said "Step 4: Total selected = 3+4+3 = 9 students" earlier. 
        # Actually 3+4+3=10. So step 4 is correct.
        # Error: Step 5: 28+32+25=75 (WRONG — should be 85). 
        # Step 6 follows from wrong Step 5.
        # Correct: total=85, percentage=10/85*100=11.8%
        "answer": {"correct_total": "85", "error_step": "5"},
        "scoring": "error_detect_v2",
    },
    {
        "id": "ood_v2_10",
        "type": "mcc_error_detect",
        "prompt": (
            "A student calculated: '15% of 840 = 12.6'\n\n"
            "Is this correct? If not, identify the error and give the correct answer."
        ),
        # Error: 15% of 840 = 840 * 0.15 = 126 (student wrote 12.6 — off by factor of 10)
        "answer": {"correct_value": "126"},
        "scoring": "error_detect_v2",
    },
]


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_ood_state_update(response: str, answer: dict) -> float:
    """Fraction of entity counts correctly found in response."""
    hits = 0
    for entity, val in answer.items():
        val_str = str(val)
        if val_str in response:
            hits += 1
    return hits / len(answer)


def score_ood_error_detect_v2(task: dict, response: str) -> float:
    """Deterministic scoring for embedded-error MCC tasks."""
    answer = task["answer"]
    task_id = task["id"]
    resp_lower = response.lower()

    if task_id == "ood_v2_07":
        # Must identify 10.80 as correct AND give correct change $9.20
        has_key = "10.80" in response or "10.8" in response
        has_final = "9.20" in response or "9.2" in response
        return 1.0 if (has_key and has_final) else (0.5 if (has_key or has_final) else 0.0)

    elif task_id == "ood_v2_08":
        # Must identify ordering issue: sent before confirming
        key_words = ["before confirm", "before getting confirm", "prior to confirm",
                     "should have confirmed", "confirm first", "confirmation before",
                     "step 3 before step 4", "sent before"]
        has_order_issue = any(kw in resp_lower for kw in key_words)
        # Also accept if response explicitly says Step 3 is wrong OR mentions the correct order
        has_correct_order = ("confirm" in resp_lower and
                             ("first" in resp_lower or "before send" in resp_lower or "then send" in resp_lower))
        return 1.0 if (has_order_issue or has_correct_order) else 0.0

    elif task_id == "ood_v2_09":
        # Must identify 85 as correct total (28+32+25=85)
        return 1.0 if "85" in response else 0.0

    elif task_id == "ood_v2_10":
        # Must identify 126 as correct answer
        return 1.0 if "126" in response else 0.0

    return 0.0


def score_ood_task_v2(task: dict, response: str) -> float:
    t = task["type"]
    scoring = task.get("scoring", "")
    if t == "wmf_state_update":
        return score_ood_state_update(response, task["answer"])
    elif scoring == "error_detect_v2":
        return score_ood_error_detect_v2(task, response)
    return 0.0


# ── OOD v2 runner ────────────────────────────────────────────────────────────

def run_ood_tasks_v2(model_name: str) -> dict:
    scores = []
    results = []
    for task in OOD_TASKS_V2:
        try:
            resp = call_model(
                model_name, task["prompt"],
                system="You are a careful, precise assistant. Think step by step.",
            )
            score = score_ood_task_v2(task, resp)
            scores.append(score)
            results.append({
                "id": task["id"], "type": task["type"],
                "score": score, "response": resp[:300],
            })
        except Exception as e:
            scores.append(0.0)
            results.append({"id": task["id"], "error": str(e), "score": 0.0})

    wmf_scores = [s for t, s in zip(OOD_TASKS_V2, scores) if t["type"] == "wmf_state_update"]
    mcc_scores = [s for t, s in zip(OOD_TASKS_V2, scores) if t["type"] == "mcc_error_detect"]

    return {
        "mean_score": float(np.mean(scores)),
        "n_tasks": len(OOD_TASKS_V2),
        "by_task": results,
        "wmf_ood": float(np.mean(wmf_scores)) if wmf_scores else 0.0,
        "mcc_ood": float(np.mean(mcc_scores)) if mcc_scores else 0.0,
    }


# ── Main per-model runner ────────────────────────────────────────────────────

def run_model_v2(model_name: str, n_mcc_problems: int = 20) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"{'='*60}", flush=True)

    result = {"model": model_name, "timestamp": datetime.utcnow().isoformat()}

    t0 = time.time()
    print("  [1/4] Completion tasks...", flush=True)
    result["completion"] = run_completion(model_name)
    print(f"        acc={result['completion']['accuracy']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    print("  [2/4] WMF-AM...", flush=True)
    result["wmf_am"] = run_wmf_am_compact(model_name)
    print(f"        mean={result['wmf_am']['mean_accuracy']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    print("  [3/4] MCC (MA + CE)...", flush=True)
    problems = load_problems(n=n_mcc_problems)
    ma_results = run_mcc_ma(model_name, problems, batch_size=n_mcc_problems)
    ce_results = run_mcc_ce(model_name, problems[:15], batch_size=15)
    mcc_ma = compute_monitoring_accuracy(ma_results)
    mcc_ce_dict = compute_control_efficacy(ce_results)
    result["mcc_ma"] = float(mcc_ma) if not np.isnan(mcc_ma) else 0.0
    result["mcc_ce"] = float(mcc_ce_dict.get("correction_efficacy", 0.0))
    result["mcc_flagging_rate"] = float(mcc_ce_dict.get("flagging_rate", 0.0))
    result["mcc_false_alarm"] = float(mcc_ce_dict.get("false_alarm_rate", 0.0))
    result["mcc_quadrant"] = (
        "HighMA_HighCE" if result["mcc_ma"] >= 0.4 and result["mcc_ce"] >= 0.4 else
        "HighMA_LowCE" if result["mcc_ma"] >= 0.4 else
        "LowMA_HighCE" if result["mcc_ce"] >= 0.4 else
        "LowMA_LowCE"
    )
    result["mcc_composite"] = 0.55 * result["mcc_ma"] + 0.45 * result["mcc_ce"]
    print(f"        MA={result['mcc_ma']:.3f}  CE={result['mcc_ce']:.3f}  quad={result['mcc_quadrant']}  ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    print("  [4/4] OOD tasks v2 (hard)...", flush=True)
    result["ood"] = run_ood_tasks_v2(model_name)
    print(f"        ood={result['ood']['mean_score']:.3f}  wmf_ood={result['ood']['wmf_ood']:.3f}  mcc_ood={result['ood']['mcc_ood']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    wmf_score = result["wmf_am"]["mean_accuracy"]
    ood_wmf = result["ood"]["wmf_ood"]
    result["cef_sentinel"] = 0.35 * wmf_score + 0.40 * result["mcc_composite"] + 0.25 * ood_wmf

    print(f"  CEF-sentinel={result['cef_sentinel']:.3f}  completion={result['completion']['accuracy']:.3f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="CEF Dissociation Pilot v2 — Hard OOD")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    selected = args.models or [
        "ollama:qwen2.5:7b",
        "ollama:qwen2.5:14b",
        "ollama:llama3.1:8b",
    ]

    print(f"CEF Dissociation Pilot v2 — {len(selected)} models", flush=True)
    print(f"Models: {selected}", flush=True)
    print(f"OOD: {len(OOD_TASKS_V2)} hard tasks (K=7-9 + embedded errors)", flush=True)
    print(f"Started: {datetime.utcnow().isoformat()}", flush=True)

    all_results = []
    for model in selected:
        try:
            r = run_model_v2(model)
            all_results.append(r)
        except Exception as e:
            print(f"ERROR on {model}: {e}", flush=True)

    dissociation = analyze_dissociation(all_results) if len(all_results) >= 2 else {}

    # Print summary
    print("\n\n" + "="*70, flush=True)
    print("DISSOCIATION PILOT v2 RESULTS", flush=True)
    print("="*70, flush=True)
    print(f"\n{'Model':<20} {'Compl':>6} {'WMF-AM':>8} {'MCC-MA':>8} {'CEF':>8} {'OOD':>8} {'WMF-OOD':>8} {'MCC-OOD':>8}", flush=True)
    print("-"*85, flush=True)
    for r in all_results:
        model = r["model"].replace("ollama:","")
        print(
            f"{model:<20} {r['completion']['accuracy']:>6.3f} "
            f"{r['wmf_am']['mean_accuracy']:>8.3f} {r['mcc_ma']:>8.3f} "
            f"{r['cef_sentinel']:>8.3f} {r['ood']['mean_score']:>8.3f} "
            f"{r['ood']['wmf_ood']:>8.3f} {r['ood']['mcc_ood']:>8.3f}",
            flush=True,
        )

    if dissociation.get("dissociation_pairs"):
        print(f"\nDissociation pairs ({dissociation['n_dissociation_pairs']}):", flush=True)
        for p in dissociation["dissociation_pairs"]:
            print(f"  {p['model_a']} vs {p['model_b']}: "
                  f"Δcompletion={p['completion_gap']:.3f} Δcef={p['cef_gap']:.3f} "
                  f"Δood={p['ood_gap']:.3f} cef_predicts_ood={p['direction_correct']}", flush=True)

    tau_cef = dissociation.get("tau_cef_ood", float("nan"))
    tau_comp = dissociation.get("tau_completion_ood", float("nan"))
    print(f"\nRank correlations (N={len(all_results)} models):", flush=True)
    print(f"  tau(CEF, OOD)        = {tau_cef}  p={dissociation.get('p_cef_ood', float('nan'))}", flush=True)
    print(f"  tau(Completion, OOD) = {tau_comp}  p={dissociation.get('p_completion_ood', float('nan'))}", flush=True)
    print(f"  CEF beats completion? {dissociation.get('cef_beats_completion', '?')}", flush=True)

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": "v2",
        "models": selected,
        "n_models": len(all_results),
        "per_model": all_results,
        "dissociation": dissociation,
    }
    out_path = args.output or str(RESULTS_DIR / "cef_dissociation_v2.json")
    Path(out_path).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}", flush=True)
    print(f"Finished: {datetime.utcnow().isoformat()}", flush=True)


if __name__ == "__main__":
    main()
