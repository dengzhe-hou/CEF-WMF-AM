"""
Experiment E4: Episodic Memory Coherence (EMC)

Three sub-dimensions:
  EMC-TO  Temporal Ordering           (reconstruct sequence of events)
  EMC-SA  Source Attribution Accuracy (attribute facts to correct sources)
  EMC-EI  Episodic Interference       (discriminate between similar episodes)

Usage:
    python episodic_discrimination.py --model gpt-4o --sub-dim all
"""

import argparse
import json
import random
import re
import time

import numpy as np
from scipy.stats import kendalltau

from config import (
    EMC_EPISODE_COUNTS,
    EMC_EVENT_COUNTS,
    EMC_SOURCE_COUNTS,
    MODELS,
    RESULTS_DIR,
    call_model,
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ── EMC-TO: Temporal Ordering ────────────────────────────────────────────────

EVENT_TEMPLATES = [
    ("started the project", "January"),
    ("hired the first engineer", "February"),
    ("launched the beta version", "March"),
    ("secured Series A funding", "April"),
    ("expanded to a second office", "May"),
    ("released version 2.0", "June"),
    ("partnered with a major client", "July"),
    ("won an industry award", "August"),
    ("opened international operations", "September"),
    ("went public on the stock exchange", "October"),
    ("reached one million users", "November"),
    ("acquired a competitor", "December"),
]


def build_temporal_ordering_prompt(n_events: int) -> tuple[str, list]:
    """Build a multi-turn conversation with N events; return prompt and true order."""
    events = random.sample(EVENT_TEMPLATES, n_events)
    # Shuffle presentation order (not chronological)
    shuffled = events[:]
    random.shuffle(shuffled)

    conversation_lines = []
    for month, (event, _) in enumerate(shuffled):
        turn_num = month + 1
        conversation_lines.append(
            f"[Turn {turn_num}] The user mentioned they {event}."
        )

    conversation = "\n".join(conversation_lines)

    # Create the true chronological order based on month index
    true_order = sorted(events, key=lambda e: EVENT_TEMPLATES.index(e))
    true_order_events = [e[0] for e in true_order]

    query_events = [e[0] for e in events]
    query_str = "\n".join(f"{i+1}. {e}" for i, e in enumerate(query_events))

    prompt = f"""Below is a conversation log. Events happened in the months listed.

{conversation}

Each event was associated with a month when mentioned. Based only on the months mentioned, put these events in chronological order (earliest to latest):

{query_str}

Respond with ONLY the reordered event numbers, e.g.: "3, 1, 4, 2, ..." """

    return prompt, true_order_events, query_events


def _parse_ordering(response: str, n: int) -> list[int]:
    """Parse a comma-separated ordering from model response."""
    nums = re.findall(r"\d+", response)
    order = [int(n) for n in nums if 1 <= int(n) <= n]
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for x in order:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    # Fill missing positions
    missing = [i for i in range(1, n + 1) if i not in seen]
    return (deduped + missing)[:n]


def run_emc_to(model_name: str, n_runs: int = 20) -> list[dict]:
    """Run EMC-TO across event count conditions."""
    results = []
    for n in EMC_EVENT_COUNTS:
        for _ in range(n_runs):
            prompt, true_order, query_events = build_temporal_ordering_prompt(n)
            response = call_model(model_name, prompt)

            predicted_order_indices = _parse_ordering(response, n)
            # Convert to ranking: what rank did each query event get?
            predicted_ranks = [0] * n
            for rank, idx in enumerate(predicted_order_indices):
                if 1 <= idx <= n:
                    predicted_ranks[idx - 1] = rank

            # True ranks (by chronological order of true_order)
            true_ranks = [0] * n
            for rank, event in enumerate(true_order):
                if event in query_events:
                    idx = query_events.index(event)
                    true_ranks[idx] = rank

            tau, p_value = kendalltau(true_ranks, predicted_ranks)
            results.append({
                "sub_dim": "EMC-TO",
                "model": model_name,
                "n_events": n,
                "kendall_tau": round(float(tau), 4),
                "p_value": round(float(p_value), 4),
            })
            time.sleep(0.5)
    return results


# ── EMC-SA: Source Attribution ───────────────────────────────────────────────

SOURCE_TEMPLATES = {
    "Alice_email": [
        ("meeting time", "3pm on Thursday"),
        ("conference room", "Room 404"),
        ("dress code", "business casual"),
    ],
    "Bob_report": [
        ("budget", "$75,000"),
        ("timeline", "6 months"),
        ("team size", "8 engineers"),
    ],
    "Carol_memo": [
        ("deadline", "March 31"),
        ("priority", "high"),
        ("stakeholder", "the CTO"),
    ],
    "website": [
        ("office address", "123 Innovation Drive"),
        ("phone number", "(555) 867-5309"),
        ("support email", "help@company.com"),
    ],
    "slack_message": [
        ("lunch meeting", "noon at the cafeteria"),
        ("parking spot", "Level B, Spot 42"),
        ("wifi password", "GreenApple2026"),
    ],
}


def build_source_attribution_prompt(n_sources: int) -> tuple[str, list]:
    """Build a multi-source context with attribution queries."""
    source_names = random.sample(list(SOURCE_TEMPLATES.keys()), n_sources)
    context_parts = []
    all_facts = []

    for source in source_names:
        facts = SOURCE_TEMPLATES[source]
        source_display = source.replace("_", " ").title()
        context_parts.append(f"[From {source_display}]:")
        for fact_key, fact_val in facts[:2]:  # use 2 facts per source
            context_parts.append(f"  - {fact_key}: {fact_val}")
            all_facts.append({
                "source": source_display,
                "key": fact_key,
                "value": fact_val,
            })

    context = "\n".join(context_parts)

    # Query 3 random facts
    query_facts = random.sample(all_facts, min(3, len(all_facts)))
    queries = "\n".join(
        f"{i+1}. Which source mentioned the {f['key']}?"
        for i, f in enumerate(query_facts)
    )

    prompt = f"""Read the following information from multiple sources, then answer the attribution questions.

{context}

[Filler text to increase distance:]
The quarterly review will be held in the main conference center. All departments are expected to present their progress reports before the end of the week.

Questions — identify the EXACT SOURCE for each piece of information:
{queries}

Format your response:
1. [source name]
2. [source name]
3. [source name]"""

    return prompt, query_facts


def run_emc_sa(model_name: str, n_runs: int = 20) -> list[dict]:
    """Run EMC-SA across source count conditions."""
    results = []
    for n_sources in EMC_SOURCE_COUNTS:
        for _ in range(n_runs):
            prompt, query_facts = build_source_attribution_prompt(n_sources)
            response = call_model(model_name, prompt)

            # Parse responses
            response_lines = [l.strip() for l in response.split("\n") if re.match(r"^\d+\.", l.strip())]

            for i, fact in enumerate(query_facts):
                if i < len(response_lines):
                    predicted_source = response_lines[i].lstrip("0123456789. ").strip()
                    correct_source = fact["source"]
                    is_correct = correct_source.lower() in predicted_source.lower()
                else:
                    predicted_source = ""
                    is_correct = False

                results.append({
                    "sub_dim": "EMC-SA",
                    "model": model_name,
                    "n_sources": n_sources,
                    "fact_key": fact["key"],
                    "correct_source": fact["source"],
                    "predicted_source": predicted_source,
                    "is_correct": int(is_correct),
                })
            time.sleep(0.5)
    return results


# ── EMC-EI: Episodic Interference ────────────────────────────────────────────

EPISODE_PAIRS = [
    {
        "episode_a": {
            "label": "Meeting with TechCorp",
            "details": {
                "budget": "$45,000",
                "deadline": "February 15",
                "contact": "Sarah Johnson",
                "outcome": "contract signed",
            },
        },
        "episode_b": {
            "label": "Meeting with DataFlow Inc",
            "details": {
                "budget": "$80,000",
                "deadline": "April 30",
                "contact": "Mark Williams",
                "outcome": "proposal pending",
            },
        },
        "queries": [
            {"q": "What was the budget discussed in the TechCorp meeting?", "answer": "$45,000", "source": "a"},
            {"q": "Who was the contact person at DataFlow Inc?", "answer": "Mark Williams", "source": "b"},
            {"q": "What was the deadline set in the TechCorp meeting?", "answer": "February 15", "source": "a"},
            {"q": "What was the outcome of the DataFlow Inc meeting?", "answer": "proposal pending", "source": "b"},
        ],
    },
    {
        "episode_a": {
            "label": "Project Alpha kickoff",
            "details": {
                "team_size": "6 developers",
                "duration": "3 months",
                "tech_stack": "Python and React",
                "client": "Horizon Bank",
            },
        },
        "episode_b": {
            "label": "Project Beta kickoff",
            "details": {
                "team_size": "10 developers",
                "duration": "6 months",
                "tech_stack": "Java and Angular",
                "client": "Summit Insurance",
            },
        },
        "queries": [
            {"q": "How many developers were assigned to Project Alpha?", "answer": "6 developers", "source": "a"},
            {"q": "What was the duration of Project Beta?", "answer": "6 months", "source": "b"},
            {"q": "What tech stack was used in Project Alpha?", "answer": "Python and React", "source": "a"},
            {"q": "Who was the client for Project Beta?", "answer": "Summit Insurance", "source": "b"},
        ],
    },
]


def build_interference_prompt(episode_pair: dict, n_interfering: int) -> tuple[str, list]:
    """Build episode interference test with N episodes."""
    epi_a = episode_pair["episode_a"]
    epi_b = episode_pair["episode_b"]

    context_a = f"[{epi_a['label']}]\n" + "\n".join(f"  - {k}: {v}" for k, v in epi_a["details"].items())
    context_b = f"[{epi_b['label']}]\n" + "\n".join(f"  - {k}: {v}" for k, v in epi_b["details"].items())

    # Add N-1 interference episodes (similar structure, random values)
    interference = ""
    if n_interfering > 1:
        extra_episodes = []
        for j in range(2, n_interfering + 1):
            extra = f"[Project Gamma-{j} kickoff]\n  - team_size: {random.randint(4, 12)} developers\n  - duration: {random.randint(2, 9)} months"
            extra_episodes.append(extra)
        interference = "\n\n".join(extra_episodes) + "\n\n"

    context = f"{context_a}\n\n{interference}{context_b}"

    queries = episode_pair["queries"]
    query_str = "\n".join(f"{i+1}. {q['q']}" for i, q in enumerate(queries))

    prompt = f"""You will read records from multiple meetings or projects. Answer questions accurately, distinguishing between them.

{context}

Questions:
{query_str}

Format:
1. [answer]
2. [answer]
3. [answer]
4. [answer]"""

    return prompt, queries


def run_emc_ei(model_name: str, n_runs: int = 10) -> list[dict]:
    """Run EMC-EI across episode count conditions."""
    results = []
    for n_episodes in EMC_EPISODE_COUNTS:
        for pair in EPISODE_PAIRS:
            for _ in range(n_runs):
                prompt, queries = build_interference_prompt(pair, n_episodes)
                response = call_model(model_name, prompt)

                # Parse answers
                response_lines = [l.strip() for l in response.split("\n") if re.match(r"^\d+\.", l.strip())]

                for i, q in enumerate(queries):
                    if i < len(response_lines):
                        predicted = response_lines[i].lstrip("0123456789. ").strip()
                        is_correct = q["answer"].lower() in predicted.lower()
                    else:
                        predicted = ""
                        is_correct = False

                    results.append({
                        "sub_dim": "EMC-EI",
                        "model": model_name,
                        "n_interfering_episodes": n_episodes,
                        "question": q["q"][:60],
                        "correct_answer": q["answer"],
                        "predicted": predicted,
                        "is_correct": int(is_correct),
                        "source_episode": q["source"],
                    })
                time.sleep(0.5)
    return results


# ── Composite EMC Score ──────────────────────────────────────────────────────

def compute_emc_score(to_results, sa_results, ei_results) -> dict:
    """Compute EMC composite score."""
    tau_mean = np.mean([r["kendall_tau"] for r in to_results]) if to_results else 0.0
    # Kendall tau in [-1, 1]; normalize to [0, 1]
    emc_to = (float(tau_mean) + 1) / 2

    emc_sa = np.mean([r["is_correct"] for r in sa_results]) if sa_results else 0.0

    emc_ei = np.mean([r["is_correct"] for r in ei_results]) if ei_results else 0.0

    composite = 0.30 * float(emc_to) + 0.40 * float(emc_sa) + 0.30 * float(emc_ei)

    # Interference coefficient: accuracy at 1 episode vs. max episodes
    if ei_results:
        acc_1 = np.mean([r["is_correct"] for r in ei_results if r["n_interfering_episodes"] == min(EMC_EPISODE_COUNTS)])
        acc_max = np.mean([r["is_correct"] for r in ei_results if r["n_interfering_episodes"] == max(EMC_EPISODE_COUNTS)])
        interference_coefficient = float(acc_1) - float(acc_max)
    else:
        interference_coefficient = 0.0

    return {
        "EMC-TO (tau_normalized)": round(float(emc_to), 4),
        "EMC-SA": round(float(emc_sa), 4),
        "EMC-EI": round(float(emc_ei), 4),
        "EMC_composite": round(composite, 4),
        "interference_coefficient": round(interference_coefficient, 4),
    }


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run EMC experiments.")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--sub-dim", default="all", choices=["all", "to", "sa", "ei"])
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args()

    to_results, sa_results, ei_results = [], [], []

    if args.sub_dim in ("all", "to"):
        print(f"Running EMC-TO for {args.model}...")
        to_results = run_emc_to(args.model, args.n_runs)

    if args.sub_dim in ("all", "sa"):
        print(f"Running EMC-SA for {args.model}...")
        sa_results = run_emc_sa(args.model, args.n_runs)

    if args.sub_dim in ("all", "ei"):
        print(f"Running EMC-EI for {args.model}...")
        ei_results = run_emc_ei(args.model, args.n_runs)

    scores = compute_emc_score(to_results, sa_results, ei_results)

    out_dir = RESULTS_DIR / "emc" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("to", to_results), ("sa", sa_results), ("ei", ei_results)]:
        if data:
            with open(out_dir / f"{name}_results.jsonl", "w") as f:
                for r in data:
                    f.write(json.dumps(r) + "\n")

    with open(out_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\nEMC Results for {args.model}:")
    for k, v in scores.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
