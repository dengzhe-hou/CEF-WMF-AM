"""
oos_validation.py — Out-of-Sample Predictive Validation

Runs WMF-AM probe AND agent battery on ONE held-out model (llama3.1:70b)
that was never part of the 20-model study set.

Protocol:
  1. WMF-AM: K=3/5/7, 4 seeds (2026,100,200,300), 5 probes/depth/seed = 20 per depth.
     Mean accuracy across all depths = WMF-AM score.
  2. Agent battery: same 10-task deterministic ReAct battery used in the main study.
     Aggregate completion rate = agent score.
  3. Rank check: place the held-out model in the N=20 ranking by WMF-AM,
     verify it lands within a plausible band of its agent-score rank.
  4. Kendall τ with N=21 (20 study + 1 held-out).

Purpose: address reviewer concern that predictive validity τ=0.612 is in-sample only.

Usage:
    python oos_validation.py
"""

import json
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import call_model, RESULTS_DIR

# ── Config ────────────────────────────────────────────────────────────────────

HELD_OUT_MODEL = "ollama:llama3.1:70b"
SEEDS = [2026, 100, 200, 300]
DEPTHS = [3, 5, 7]
PROBES_PER_DEPTH_PER_SEED = 5  # 4 seeds × 5 probes = 20 per depth
WMF_CALL_TIMEOUT = 120  # seconds

# The 20-model study results (from cef_predictive_validity.json + paper tables)
STUDY_RESULTS = {
    "ollama:deepseek-r1:14b": {"wmf_am": 0.983, "agent": 0.70},
    "ollama:qwen2.5:32b":     {"wmf_am": 0.650, "agent": 0.90},
    "ollama:qwen2.5:14b":     {"wmf_am": 0.467, "agent": 0.90},
    "ollama:gemma2:27b":      {"wmf_am": 0.450, "agent": 0.80},
    "ollama:qwen2.5:7b":      {"wmf_am": 0.350, "agent": 0.90},
    "ollama:mistral:7b":      {"wmf_am": 0.350, "agent": 0.30},
    "ollama:llama3.1:8b":     {"wmf_am": 0.183, "agent": 0.60},
    "ollama:phi3:14b":        {"wmf_am": 0.267, "agent": 0.20},
    "ollama:gemma2:9b":       {"wmf_am": 0.400, "agent": 0.90},
    "ollama:qwen2.5:3b":      {"wmf_am": 0.200, "agent": 0.40},
    "ollama:llama3.2:3b":     {"wmf_am": 0.133, "agent": 0.30},
    "ollama:deepseek-r1:7b":  {"wmf_am": 0.150, "agent": 0.40},
    "ollama:mixtral:8x7b":    {"wmf_am": 0.300, "agent": 0.40},
    "ollama:command-r:35b":   {"wmf_am": 0.350, "agent": 0.70},
    "ollama:yi:34b":          {"wmf_am": 0.250, "agent": 0.30},
    "ollama:gemma2:2b":       {"wmf_am": 0.217, "agent": 0.40},
    "ollama:qwen2.5:1.5b":    {"wmf_am": 0.117, "agent": 0.30},
    "ollama:tinyllama:1.1b":  {"wmf_am": 0.117, "agent": 0.00},
    "ollama:llama3.2:1b":     {"wmf_am": 0.067, "agent": 0.20},
    "ollama:qwen2.5:0.5b":    {"wmf_am": 0.050, "agent": 0.00},
}


# ── WMF-AM Probe ──────────────────────────────────────────────────────────────

def build_wmf_am_problem(k_operations: int, seed: int, probe_idx: int):
    """Generate a WMF-AM problem identical to main study protocol."""
    rng = random.Random(seed + k_operations * 1000 + probe_idx)

    ENTITY_TEMPLATES = [
        ("Alice", "owns", "paintings"), ("Bob", "has", "coins"),
        ("Carol", "collected", "stamps"), ("David", "saved", "documents"),
        ("Emma", "scored", "points"), ("Frank", "planted", "trees"),
        ("Grace", "wrote", "poems"), ("Henry", "built", "models"),
        ("Iris", "caught", "fish"), ("James", "sold", "tickets"),
        ("Kate", "baked", "loaves"), ("Leo", "ran", "miles"),
        ("Mia", "read", "books"), ("Noah", "drew", "sketches"),
    ]
    entities = rng.sample([e for e, _, _ in ENTITY_TEMPLATES], 3)
    state = {e: rng.randint(5, 20) for e in entities}
    initial_state = dict(state)
    operations = []

    for _ in range(k_operations):
        op_type = rng.choice(["add", "subtract", "transfer"])
        if op_type == "add":
            e = rng.choice(entities)
            amount = rng.randint(1, 10)
            state[e] += amount
            operations.append(f"{e} gains {amount} points.")
        elif op_type == "subtract":
            e = rng.choice(entities)
            amount = min(rng.randint(1, 5), state[e] - 1)
            if amount > 0:
                state[e] -= amount
                operations.append(f"{e} loses {amount} points.")
            else:
                state[e] += 1
                operations.append(f"{e} gains 1 point.")
        else:
            giver, receiver = rng.sample(entities, 2)
            amount = min(rng.randint(1, 3), state[giver] - 1)
            if amount > 0:
                state[giver] -= amount
                state[receiver] += amount
                operations.append(f"{giver} gives {amount} points to {receiver}.")
            else:
                operations.append("No transfer occurs this round.")

    query_entity = rng.choice(entities)
    return initial_state, operations, state[query_entity], query_entity


def run_wmf_am(model_key: str) -> dict:
    """Run WMF-AM on a single model using main-study protocol."""
    print(f"\n[WMF-AM] Running {model_key}...")
    by_depth = {k: [] for k in DEPTHS}
    all_trials = []

    for seed in SEEDS:
        for k in DEPTHS:
            for i in range(PROBES_PER_DEPTH_PER_SEED):
                initial_state, operations, correct, query_entity = \
                    build_wmf_am_problem(k, seed, i)

                state_str = ", ".join(f"{e}: {v} points" for e, v in initial_state.items())
                ops_str = "\n".join(f"  {j+1}. {op}" for j, op in enumerate(operations))
                prompt = (
                    "You will track a sequence of point updates. "
                    "You cannot refer back to the initial state after reading it once.\n\n"
                    f"Initial state:\n{state_str}\n\n"
                    f"Operations (apply in order):\n{ops_str}\n\n"
                    f"After all operations, how many points does {query_entity} have?\n\n"
                    "Respond with ONLY the final number."
                )

                try:
                    response = call_model(model_key, prompt)
                    nums = re.findall(r"\d+", response)
                    predicted = int(nums[0]) if nums else -1
                    accurate = int(predicted == correct)
                except Exception as e:
                    print(f"  ERROR k={k} seed={seed} i={i}: {e}")
                    predicted = -1
                    accurate = 0

                by_depth[k].append(accurate)
                all_trials.append({
                    "k": k, "seed": seed, "probe_idx": i,
                    "correct": correct, "predicted": predicted, "accurate": accurate,
                })
                print("." if accurate else "x", end="", flush=True)

    print()
    depth_means = {k: round(sum(v)/len(v), 4) if v else 0.0 for k, v in by_depth.items()}
    all_acc = [t["accurate"] for t in all_trials]
    wmf_am_score = round(sum(all_acc) / len(all_acc), 4) if all_acc else 0.0

    print(f"  WMF-AM: depth={depth_means}, overall={wmf_am_score:.4f}")
    return {"wmf_am_score": wmf_am_score, "by_depth": depth_means, "n_trials": len(all_trials)}


# ── Agent Battery ─────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are a ReAct agent. You solve tasks by interleaving Thought and Action steps.

Format — you MUST follow this EXACTLY on every turn:

THOUGHT: <your reasoning about what to do next>
ACTION: <tool_name>(<arguments>)

When you have the final answer, use:
THOUGHT: <your final reasoning>
ACTION: FINISH(<your final answer>)

Rules:
- Each response must contain exactly ONE THOUGHT and ONE ACTION.
- Do NOT include OBSERVATION — the system provides that after your ACTION.
- Available tools are listed in the task description.
- Use FINISH() only when you are confident in your answer.
"""


def parse_action(response: str):
    match = re.search(r"ACTION:\s*(\w+)\((.*)?\)", response, re.DOTALL)
    if match:
        return match.group(1).strip(), (match.group(2) or "").strip()
    match = re.search(r"(\w+)\(([^)]*)\)", response)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "INVALID", response


def run_agent(model, task_prompt, tools, max_steps=15):
    history = []
    steps = []
    errors = 0
    final_answer = None

    for step_i in range(max_steps):
        user_msg = task_prompt if step_i == 0 else steps[-1]["observation"]
        try:
            response = call_model(model, prompt=user_msg,
                                  system=REACT_SYSTEM_PROMPT,
                                  history=history if step_i > 0 else None)
        except Exception as e:
            errors += 1
            steps.append({"step": step_i, "error": str(e)})
            break

        tool_name, args_str = parse_action(response)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": response})

        if tool_name == "FINISH":
            final_answer = args_str
            steps.append({"step": step_i, "tool": "FINISH", "args": args_str})
            break
        elif tool_name == "INVALID":
            errors += 1
            steps.append({"step": step_i, "tool": "INVALID", "observation":
                          "OBSERVATION: Invalid action format. Use ACTION: tool_name(args)"})
        elif tool_name in tools:
            try:
                result = tools[tool_name](args_str)
                observation = f"OBSERVATION: {result}"
            except Exception as e:
                errors += 1
                observation = f"OBSERVATION: Error calling {tool_name}: {e}"
            steps.append({"step": step_i, "tool": tool_name, "args": args_str,
                          "observation": observation})
        else:
            errors += 1
            steps.append({"step": step_i, "tool": tool_name,
                          "observation": f"OBSERVATION: Unknown tool '{tool_name}'"})

    return {"steps": steps, "final_answer": final_answer,
            "n_steps": len(steps), "errors": errors}


# ── 10-task Agent Battery (identical to main study) ───────────────────────────

def get_tasks():
    """Return the same 10 deterministic tasks used in cef_agent_validation.py."""

    def _calc_tools():
        calc_state = {"values": {}}
        def calculator(expr):
            try:
                expr_clean = expr.strip().strip('"').strip("'")
                result = eval(expr_clean, {"__builtins__": {}}, {})
                return str(result)
            except Exception as e:
                return f"Error: {e}"
        def store(args):
            parts = args.split(",", 1)
            if len(parts) == 2:
                key = parts[0].strip().strip('"\'')
                val = parts[1].strip()
                try:
                    calc_state["values"][key] = eval(val, {"__builtins__": {}}, {})
                    return f"Stored {key} = {calc_state['values'][key]}"
                except:
                    calc_state["values"][key] = val
                    return f"Stored {key} = {val}"
            return "Error: use store(name, value)"
        def retrieve(key):
            k = key.strip().strip('"\'')
            if k in calc_state["values"]:
                return str(calc_state["values"][k])
            return f"Key '{k}' not found"
        return {"calculator": calculator, "store": store, "retrieve": retrieve}

    def _entity_tools():
        entity_state = {
            "Alice": 15, "Bob": 8, "Carol": 12, "David": 20, "Emma": 5
        }
        def get_score(name):
            n = name.strip().strip('"\'')
            if n in entity_state:
                return f"{n} has {entity_state[n]} points"
            return f"Unknown entity: {n}"
        def update_score(args):
            parts = args.split(",")
            if len(parts) == 2:
                n = parts[0].strip().strip('"\'')
                try:
                    delta = int(parts[1].strip())
                    if n in entity_state:
                        entity_state[n] += delta
                        return f"{n} now has {entity_state[n]} points"
                    return f"Unknown: {n}"
                except:
                    return "Error: use update_score(name, delta)"
            return "Error: use update_score(name, delta)"
        return {"get_score": get_score, "update_score": update_score}

    def _search_tools():
        db = {"file_001": "revenue=1200", "file_002": "cost=800",
              "file_003": "profit=400", "file_004": "tax=80", "file_005": "net=320"}
        def list_files(_=""):
            return "Files: " + ", ".join(db.keys())
        def read_file(name):
            n = name.strip().strip('"\'')
            return db.get(n, f"File not found: {n}")
        return {"list_files": list_files, "read_file": read_file}

    def _lookup_tools():
        facts = {
            "population of France": "67 million",
            "capital of Australia": "Canberra",
            "boiling point of water": "100 degrees Celsius",
            "speed of light": "approximately 300000 km/s",
            "atomic number of gold": "79",
        }
        def verify_fact(claim):
            c = claim.strip().lower().strip('"\'')
            for key, val in facts.items():
                if key in c or any(w in c for w in key.split()):
                    return f"Fact: {key} is {val}"
            return "Fact not found in database"
        return {"verify_fact": verify_fact}

    def _conflict_tools():
        sources = {
            "source_A": {"temperature": "22C", "humidity": "65%", "wind": "10 km/h"},
            "source_B": {"temperature": "24C", "humidity": "70%", "wind": "12 km/h"},
            "source_C": {"temperature": "21C", "humidity": "63%", "wind": "9 km/h"},
        }
        def query_source(src):
            s = src.strip().strip('"\'')
            if s in sources:
                return str(sources[s])
            return f"Unknown source: {s}"
        return {"query_source": query_source}

    def _recall_tools():
        conversation = [
            ("User", "I'm planning a trip to Tokyo next month."),
            ("Assistant", "That sounds exciting! Tokyo has great food and culture."),
            ("User", "I want to visit the Senso-ji temple."),
            ("Assistant", "Senso-ji is in Asakusa district, very accessible by subway."),
            ("User", "What's the best time to visit?"),
            ("Assistant", "Early morning is best to avoid crowds."),
        ]
        def recall(turn_idx):
            try:
                idx = int(turn_idx.strip()) - 1
                if 0 <= idx < len(conversation):
                    speaker, text = conversation[idx]
                    return f"Turn {idx+1}: {speaker} said: '{text}'"
                return f"Turn {turn_idx} not found (1-{len(conversation)})"
            except:
                return "Error: provide turn number"
        def search_conversation(keyword):
            kw = keyword.strip().strip('"\'').lower()
            hits = [f"Turn {i+1}: {s}: '{t}'" for i, (s, t) in enumerate(conversation)
                    if kw in t.lower()]
            return ("\n".join(hits) if hits else f"No mentions of '{keyword}'")
        return {"recall": recall, "search_conversation": search_conversation}

    def _attribution_tools():
        sources = {
            "paper_A": "Neural networks learn features hierarchically.",
            "paper_B": "Transformers use attention to model long-range dependencies.",
            "paper_C": "Reinforcement learning optimizes cumulative reward.",
            "paper_D": "Gradient descent minimizes loss via backpropagation.",
        }
        def get_source(paper_id):
            p = paper_id.strip().strip('"\'')
            return sources.get(p, f"Paper {p} not found")
        return {"get_source": get_source}

    def _shopping_tools():
        inventory = {
            "laptop": {"price": 999, "stock": 3},
            "mouse": {"price": 29, "stock": 15},
            "keyboard": {"price": 79, "stock": 8},
            "monitor": {"price": 399, "stock": 2},
            "headphones": {"price": 149, "stock": 5},
        }
        def check_item(name):
            n = name.strip().strip('"\'').lower()
            if n in inventory:
                item = inventory[n]
                return f"{n}: ${item['price']}, {item['stock']} in stock"
            return f"Item not found: {name}"
        def check_budget(amount):
            try:
                budget = float(amount.strip())
                affordable = [f"{n} (${i['price']})" for n, i in inventory.items()
                              if i['price'] <= budget]
                return f"Items within ${budget}: {', '.join(affordable) or 'none'}"
            except:
                return "Error: provide numeric budget"
        return {"check_item": check_item, "check_budget": check_budget}

    def _schedule_tools():
        schedule = {
            "Alice": ["Mon 9-10", "Wed 14-15", "Fri 10-11"],
            "Bob": ["Mon 10-11", "Tue 9-10", "Thu 14-15"],
            "Carol": ["Tue 10-11", "Wed 9-10", "Fri 14-15"],
        }
        def check_availability(person):
            p = person.strip().strip('"\'')
            if p in schedule:
                return f"{p} is busy: {', '.join(schedule[p])}"
            return f"Person not found: {p}"
        def find_slot(day):
            d = day.strip().strip('"\'')
            available = {p: [s for s in slots if not s.startswith(d)]
                         for p, slots in schedule.items()}
            return str(available)
        return {"check_availability": check_availability, "find_slot": find_slot}

    def _pipeline_tools():
        data = {"raw": [10, 20, 15, 30, 25]}
        results = {}
        def load_data(_=""):
            return f"Loaded {len(data['raw'])} records: {data['raw']}"
        def compute_mean(_=""):
            vals = data["raw"]
            mean = sum(vals) / len(vals)
            results["mean"] = mean
            return f"Mean: {mean}"
        def compute_max(_=""):
            m = max(data["raw"])
            results["max"] = m
            return f"Max: {m}"
        def compute_sum(_=""):
            s = sum(data["raw"])
            results["sum"] = s
            return f"Sum: {s}"
        return {"load_data": load_data, "compute_mean": compute_mean,
                "compute_max": compute_max, "compute_sum": compute_sum}

    tasks = [
        {
            "id": "multi_step_calc",
            "name": "Multi-Step Calculation",
            "cef_dim": "WMF-AM",
            "prompt": (
                "Use the calculator to solve this problem step by step:\n"
                "Start with 100. Multiply by 3. Subtract 50. Divide by 5. Add 20.\n"
                "What is the final result?\n\n"
                "Available tools:\n"
                "  calculator(expression) — evaluates a math expression\n"
                "  store(name, value) — stores a value\n"
                "  retrieve(name) — retrieves a stored value\n\n"
                "Use FINISH(answer) when done."
            ),
            "tools": _calc_tools(),
            "correct_answer": "90",
            "check": lambda ans: ans is not None and "90" in str(ans),
        },
        {
            "id": "entity_tracking",
            "name": "Entity Score Tracking",
            "cef_dim": "WMF-AM",
            "prompt": (
                "Track scores for 5 players. Apply these updates:\n"
                "1. Alice gains 5 points\n"
                "2. Bob loses 3 points\n"
                "3. Carol gains 8 points\n"
                "4. Alice loses 2 points\n"
                "5. David gains 10 points\n"
                "Who has the highest score after all updates?\n\n"
                "Available tools:\n"
                "  get_score(name) — get current score\n"
                "  update_score(name, delta) — add delta to score\n\n"
                "Use FINISH(name) when done."
            ),
            "tools": _entity_tools(),
            "correct_answer": "David",
            "check": lambda ans: ans is not None and "David" in str(ans),
        },
        {
            "id": "sequential_search",
            "name": "Sequential File Search",
            "cef_dim": "WMF-AM",
            "prompt": (
                "Find the net profit from financial files.\n"
                "You need to find the 'net' value.\n\n"
                "Available tools:\n"
                "  list_files() — list available files\n"
                "  read_file(name) — read a file's content\n\n"
                "Use FINISH(value) when done."
            ),
            "tools": _search_tools(),
            "correct_answer": "320",
            "check": lambda ans: ans is not None and "320" in str(ans),
        },
        {
            "id": "uncertain_lookup",
            "name": "Fact Verification",
            "cef_dim": "MCC-MA",
            "prompt": (
                "Verify: What is the capital of Australia?\n\n"
                "Available tools:\n"
                "  verify_fact(claim) — check a factual claim\n\n"
                "Use FINISH(answer) when done."
            ),
            "tools": _lookup_tools(),
            "correct_answer": "Canberra",
            "check": lambda ans: ans is not None and "Canberra" in str(ans),
        },
        {
            "id": "multi_source_conflict",
            "name": "Multi-Source Aggregation",
            "cef_dim": "MCC-MA",
            "prompt": (
                "Query all three weather sources (source_A, source_B, source_C) "
                "and report the average temperature.\n\n"
                "Available tools:\n"
                "  query_source(source_name) — query a weather source\n\n"
                "Use FINISH(average) when done."
            ),
            "tools": _conflict_tools(),
            "correct_answer": "22.33",
            "check": lambda ans: ans is not None and any(
                x in str(ans) for x in ["22", "22.3", "22.33", "22.4"]),
        },
        {
            "id": "conversation_recall",
            "name": "Conversation Memory",
            "cef_dim": "EMC",
            "prompt": (
                "Search the conversation history for mentions of 'temple'.\n"
                "What specific temple was mentioned?\n\n"
                "Available tools:\n"
                "  recall(turn_number) — read a specific turn\n"
                "  search_conversation(keyword) — search for keyword\n\n"
                "Use FINISH(temple_name) when done."
            ),
            "tools": _recall_tools(),
            "correct_answer": "Senso-ji",
            "check": lambda ans: ans is not None and "Senso" in str(ans),
        },
        {
            "id": "source_attribution",
            "name": "Source Attribution",
            "cef_dim": "EMC",
            "prompt": (
                "Which paper discusses attention mechanisms in transformers?\n\n"
                "Available tools:\n"
                "  get_source(paper_id) — get paper content (papers: paper_A through paper_D)\n\n"
                "Use FINISH(paper_id) when done."
            ),
            "tools": _attribution_tools(),
            "correct_answer": "paper_B",
            "check": lambda ans: ans is not None and "paper_B" in str(ans),
        },
        {
            "id": "shopping_assistant",
            "name": "Shopping Budget Check",
            "cef_dim": "GENERAL",
            "prompt": (
                "A customer has a $100 budget. What items can they afford?\n\n"
                "Available tools:\n"
                "  check_item(name) — check price and stock\n"
                "  check_budget(amount) — find items within budget\n\n"
                "Use FINISH(list) when done."
            ),
            "tools": _shopping_tools(),
            "correct_answer": "mouse,keyboard",
            "check": lambda ans: ans is not None and (
                "mouse" in str(ans).lower() and "keyboard" in str(ans).lower()),
        },
        {
            "id": "schedule_coordination",
            "name": "Schedule Coordination",
            "cef_dim": "GENERAL",
            "prompt": (
                "Find when Alice, Bob, and Carol are all busy on Monday.\n\n"
                "Available tools:\n"
                "  check_availability(person) — get person's busy times\n"
                "  find_slot(day) — check availability for a day\n\n"
                "Use FINISH(answer) when done."
            ),
            "tools": _schedule_tools(),
            "correct_answer": "Mon 9-10 and Mon 10-11",
            "check": lambda ans: ans is not None and "Mon" in str(ans),
        },
        {
            "id": "data_pipeline",
            "name": "Data Pipeline",
            "cef_dim": "GENERAL",
            "prompt": (
                "Process the dataset: compute the mean, max, and sum of the data.\n"
                "Report all three values.\n\n"
                "Available tools:\n"
                "  load_data() — load the dataset\n"
                "  compute_mean() — compute mean\n"
                "  compute_max() — compute maximum\n"
                "  compute_sum() — compute sum\n\n"
                "Use FINISH(mean=X, max=Y, sum=Z) when done."
            ),
            "tools": _pipeline_tools(),
            "correct_answer": "mean=20 max=30 sum=100",
            "check": lambda ans: ans is not None and all(
                x in str(ans) for x in ["20", "30", "100"]),
        },
    ]
    return tasks


def run_agent_battery(model_key: str, max_steps: int = 15) -> dict:
    """Run 10-task agent battery on one model."""
    print(f"\n[Agent Battery] Running {model_key}...")
    tasks = get_tasks()
    results = []

    for task in tasks:
        print(f"  Task {task['id']}: ", end="", flush=True)
        t0 = time.time()
        try:
            agent_result = run_agent(model_key, task["prompt"],
                                     task["tools"], max_steps)
            correct = task["check"](agent_result["final_answer"])
            elapsed = time.time() - t0
            print(f"{'PASS' if correct else 'fail'} ({elapsed:.1f}s)")
            results.append({
                "task_id": task["id"],
                "cef_dim": task["cef_dim"],
                "correct": correct,
                "final_answer": agent_result["final_answer"],
                "n_steps": agent_result["n_steps"],
                "errors": agent_result["errors"],
                "elapsed_s": round(elapsed, 1),
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR ({elapsed:.1f}s): {e}")
            results.append({
                "task_id": task["id"],
                "cef_dim": task["cef_dim"],
                "correct": False,
                "error": str(e),
            })

    n_correct = sum(1 for r in results if r["correct"])
    agent_score = n_correct / len(tasks)
    print(f"  Agent score: {n_correct}/{len(tasks)} = {agent_score:.3f}")
    return {"agent_score": agent_score, "n_correct": n_correct, "n_tasks": len(tasks),
            "task_results": results}


# ── Kendall τ + Rank Analysis ─────────────────────────────────────────────────

def compute_kendall_tau(x, y):
    """Compute Kendall tau-b."""
    import scipy.stats
    tau, p = scipy.stats.kendalltau(x, y)
    return tau, p


def rank_analysis(held_out_wmf: float, held_out_agent: float) -> dict:
    """
    Insert held-out model into the N=20 ranking and check consistency.
    Returns rank positions and concordance/discordance info.
    """
    all_models = list(STUDY_RESULTS.keys()) + [HELD_OUT_MODEL]
    all_wmf = [STUDY_RESULTS[m]["wmf_am"] for m in all_models[:-1]] + [held_out_wmf]
    all_agent = [STUDY_RESULTS[m]["agent"] for m in all_models[:-1]] + [held_out_agent]

    # Rank positions (1 = highest)
    wmf_rank = sorted(range(len(all_wmf)), key=lambda i: -all_wmf[i]).index(20) + 1
    agent_rank = sorted(range(len(all_agent)), key=lambda i: -all_agent[i]).index(20) + 1

    # τ for N=20 study
    study_wmf = [STUDY_RESULTS[m]["wmf_am"] for m in list(STUDY_RESULTS.keys())]
    study_agent = [STUDY_RESULTS[m]["agent"] for m in list(STUDY_RESULTS.keys())]
    tau_20, p_20 = compute_kendall_tau(study_wmf, study_agent)

    # τ for N=21 (including held-out)
    tau_21, p_21 = compute_kendall_tau(all_wmf, all_agent)

    print(f"\n[Rank Analysis]")
    print(f"  Held-out WMF-AM rank: {wmf_rank}/21  (WMF-AM = {held_out_wmf:.3f})")
    print(f"  Held-out Agent rank:  {agent_rank}/21 (Agent = {held_out_agent:.3f})")
    print(f"  Rank gap: |{wmf_rank} - {agent_rank}| = {abs(wmf_rank - agent_rank)}")
    print(f"\n  τ(WMF-AM, Agent) N=20 study:    {tau_20:.3f} (p={p_20:.4f})")
    print(f"  τ(WMF-AM, Agent) N=21 with OOS: {tau_21:.3f} (p={p_21:.4f})")

    return {
        "held_out_wmf_rank_21": wmf_rank,
        "held_out_agent_rank_21": agent_rank,
        "rank_gap": abs(wmf_rank - agent_rank),
        "tau_n20": round(tau_20, 4),
        "p_n20": round(p_20, 4),
        "tau_n21": round(tau_21, 4),
        "p_n21": round(p_21, 4),
        "concordant": abs(wmf_rank - agent_rank) <= 5,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"Out-of-Sample Predictive Validation")
    print(f"Held-out model: {HELD_OUT_MODEL}")
    print(f"Start: {datetime.now().isoformat()}")
    print("=" * 60)

    # Phase 1: WMF-AM
    wmf_result = run_wmf_am(HELD_OUT_MODEL)

    # Phase 2: Agent Battery
    agent_result = run_agent_battery(HELD_OUT_MODEL)

    # Phase 3: Rank / τ analysis
    rank_result = rank_analysis(wmf_result["wmf_am_score"], agent_result["agent_score"])

    # Save results
    output = {
        "protocol": "Out-of-Sample Predictive Validation",
        "held_out_model": HELD_OUT_MODEL,
        "timestamp": timestamp,
        "wmf_am": wmf_result,
        "agent_battery": agent_result,
        "rank_analysis": rank_result,
        "study_n": 20,
        "oos_n": 1,
    }
    out_path = RESULTS_DIR / f"oos_validation_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n=== Summary ===")
    print(f"Held-out model ({HELD_OUT_MODEL.split(':',1)[1]}):")
    print(f"  WMF-AM score = {wmf_result['wmf_am_score']:.3f}")
    print(f"  Agent score  = {agent_result['agent_score']:.3f}")
    print(f"  WMF rank {rank_result['held_out_wmf_rank_21']}/21, Agent rank {rank_result['held_out_agent_rank_21']}/21")
    print(f"  Rank gap = {rank_result['rank_gap']} ({'concordant' if rank_result['concordant'] else 'discordant'})")
    print(f"  τ(N=20 study) = {rank_result['tau_n20']:.3f} (p={rank_result['p_n20']:.4f})")
    print(f"  τ(N=21 +OOS)  = {rank_result['tau_n21']:.3f} (p={rank_result['p_n21']:.4f})")


if __name__ == "__main__":
    main()
