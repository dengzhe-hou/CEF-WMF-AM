#!/usr/bin/env python3
"""
Run all 4 ablation experiments on specified models.
Wraps: wmf_am_yoked_control, wmf_am_nonarithmetic, wmf_am_template_harmonization, wmf_am_control.
"""

import subprocess
import sys
import time
from datetime import datetime

# 5 small models that need ablations
SMALL_MODELS = [
    "ollama:gemma2:2b",
    "ollama:qwen2.5:1.5b",
    "ollama:tinyllama:1.1b",
    "ollama:llama3.2:1b",
    "ollama:qwen2.5:0.5b",
]

# 8 API models (7 new + llama3.1:70b)
API_MODELS = [
    "openrouter:gpt-4o",
    "openrouter:gpt-4o-mini",
    "openrouter:claude-sonnet-4",
    "openrouter:deepseek-v3",
    "openrouter:gemini-2.5-flash",
    "openrouter:o3-mini",
    "openrouter:deepseek-r1",
    "ollama:llama3.1:70b",
]

ABLATIONS = [
    ("K=1 Control", "wmf_am_control.py", "--model {model}"),
    ("Non-Arithmetic", "wmf_am_nonarithmetic.py", "--models {model}"),
    ("Yoked Control", "wmf_am_yoked_control.py", "--model {model}"),
    ("Template Harmonization", "wmf_am_template_harmonization.py", "--models {model}"),
]


def run_ablation(name, script, model_arg, model):
    cmd_arg = model_arg.format(model=model)
    cmd = f"python3 code/{script} {cmd_arg}"
    print(f"\n  [{name}] {model}")
    print(f"  CMD: {cmd}")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=3600
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            # Extract last few lines for summary
            last_lines = result.stdout.strip().split("\n")[-3:]
            print(f"  OK ({elapsed:.0f}s)")
            for line in last_lines:
                print(f"    {line}")
        else:
            print(f"  FAILED ({elapsed:.0f}s): {result.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (600s)")
    except Exception as e:
        print(f"  ERROR: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["small", "api", "all"], default="all")
    args = parser.parse_args()

    if args.group in ("small", "all"):
        models = SMALL_MODELS
    if args.group in ("api", "all"):
        models = (SMALL_MODELS if args.group == "all" else []) + API_MODELS

    print("=" * 60)
    print(f"Running all ablations on {len(models)} models")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    for model in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")
        for name, script, arg_template in ABLATIONS:
            run_ablation(name, script, arg_template, model)

    print(f"\n{'='*60}")
    print(f"All ablations complete: {datetime.now().isoformat()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
