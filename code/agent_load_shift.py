#!/usr/bin/env python3
"""
Block 5: Load-Shift Intervention on Agent Battery.

Runs agent battery in two conditions:
  1. SUPPORTED (normal): full conversation history visible
  2. UNSUPPORTED: history truncated to last 1 turn — model must carry state internally

Tests whether K_crit becomes predictive under unsupported conditions,
while WMF-AM@K=3/5/7 remains predictive under supported conditions.

Core claim: "Agent success depends on reliable state tracking at the
benchmark's operating point, not on maximum memory capacity."
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import call_model, RESULTS_DIR
from oos_validation import (
    get_tasks, parse_action, REACT_SYSTEM_PROMPT
)


def run_agent_with_history_control(
    model: str,
    task_prompt: str,
    tools: dict,
    max_steps: int = 15,
    history_window: int | None = None,
) -> dict:
    """
    Run agent with controlled history visibility.

    Args:
        history_window: None = full history (supported condition)
                       1 = only last turn visible (unsupported condition)
                       N = last N turns visible
    """
    full_history = []
    steps = []
    errors = 0
    final_answer = None

    for step_i in range(max_steps):
        user_msg = task_prompt if step_i == 0 else steps[-1].get(
            "observation", "OBSERVATION: No output from previous step.")

        # Build history based on condition
        if history_window is None:
            # SUPPORTED: full history
            visible_history = full_history.copy() if step_i > 0 else None
        else:
            # UNSUPPORTED: only last N exchanges visible
            if step_i > 0 and full_history:
                # Each exchange = 2 entries (user + assistant)
                n_entries = history_window * 2
                visible_history = full_history[-n_entries:]
            else:
                visible_history = None

        try:
            response = call_model(
                model,
                prompt=user_msg,
                system=REACT_SYSTEM_PROMPT,
                history=visible_history,
            )
        except Exception as e:
            errors += 1
            steps.append({"step": step_i, "error": str(e)})
            break

        tool_name, args_str = parse_action(response)

        # Always append to full history (for tracking), but visibility is controlled
        full_history.append({"role": "user", "content": user_msg})
        full_history.append({"role": "assistant", "content": response})

        if tool_name == "FINISH":
            final_answer = args_str
            steps.append({"step": step_i, "tool": "FINISH", "args": args_str})
            break
        elif tool_name == "INVALID":
            errors += 1
            steps.append({
                "step": step_i, "tool": "INVALID",
                "observation": "OBSERVATION: Invalid action format. Use ACTION: tool_name(args)"
            })
        elif tool_name in tools:
            try:
                result = tools[tool_name](args_str)
                observation = f"OBSERVATION: {result}"
            except Exception as e:
                errors += 1
                observation = f"OBSERVATION: Error calling {tool_name}: {e}"
            steps.append({
                "step": step_i, "tool": tool_name, "args": args_str,
                "observation": observation,
            })
        else:
            errors += 1
            steps.append({
                "step": step_i, "tool": tool_name,
                "observation": f"OBSERVATION: Unknown tool '{tool_name}'"
            })

    return {
        "steps": steps,
        "final_answer": final_answer,
        "n_steps": len(steps),
        "errors": errors,
        "history_window": history_window,
        "total_history_turns": len(full_history) // 2,
    }


def run_load_shift_battery(model: str, max_steps: int = 15) -> dict:
    """Run both supported and unsupported agent battery on one model."""
    tasks = get_tasks()
    conditions = {
        "supported": None,       # full history
        "unsupported": 1,        # only last 1 exchange visible
    }

    results = {}
    for cond_name, hist_window in conditions.items():
        print(f"\n  [{cond_name.upper()}] {model}")
        task_results = []

        for task in tasks:
            t0 = time.time()
            try:
                agent_result = run_agent_with_history_control(
                    model, task["prompt"], task["tools"],
                    max_steps=max_steps,
                    history_window=hist_window,
                )
                correct = task["check"](agent_result["final_answer"])
                elapsed = time.time() - t0
                status = "PASS" if correct else "fail"
                print(f"    {task['id']:25s} {status} ({elapsed:.1f}s)")
                task_results.append({
                    "task_id": task["id"],
                    "correct": correct,
                    "final_answer": agent_result["final_answer"],
                    "n_steps": agent_result["n_steps"],
                    "errors": agent_result["errors"],
                    "elapsed_s": round(elapsed, 1),
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f"    {task['id']:25s} ERROR ({elapsed:.1f}s): {e}")
                task_results.append({
                    "task_id": task["id"],
                    "correct": False,
                    "error": str(e),
                })

        n_correct = sum(1 for r in task_results if r["correct"])
        score = n_correct / len(tasks)
        print(f"    → Score: {n_correct}/{len(tasks)} = {score:.3f}")

        results[cond_name] = {
            "score": score,
            "n_correct": n_correct,
            "n_tasks": len(tasks),
            "history_window": hist_window,
            "task_results": task_results,
        }

    return {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "supported_score": results["supported"]["score"],
        "unsupported_score": results["unsupported"]["score"],
        "load_shift_delta": results["supported"]["score"] - results["unsupported"]["score"],
        "conditions": results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent Load-Shift Intervention")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Agent Battery Load-Shift Intervention")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Conditions: supported (full history) vs unsupported (last 1 turn)")

    all_results = []
    for model in args.models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")
        result = run_load_shift_battery(model)
        all_results.append(result)

        # Save incrementally
        out_path = args.output or str(
            RESULTS_DIR / f"load_shift_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        )
        with open(out_path, "w") as f:
            json.dump({"results": all_results}, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("LOAD-SHIFT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'Supported':>10} {'Unsupported':>12} {'Delta':>8}")
    print("-" * 68)
    for r in all_results:
        model = r["model"].replace("openrouter:", "").replace("ollama:", "")
        print(f"  {model:<33} {r['supported_score']:>10.3f} "
              f"{r['unsupported_score']:>12.3f} {r['load_shift_delta']:>+8.3f}")

    print(f"\nFinal results saved to {out_path}")


if __name__ == "__main__":
    main()
