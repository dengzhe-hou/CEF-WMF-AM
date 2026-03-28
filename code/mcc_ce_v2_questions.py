"""
MCC-CE-v2 Question Bank — Harder Questions for Control Efficacy

PURPOSE:
  The original MCC-CE produced a universal floor effect (CE=0 for all 7 models)
  because models rarely got questions wrong → nothing to flag → empty denominator.
  v2 needs questions where models get ~30-60% wrong, creating meaningful errors
  to test flagging and correction ability.

DESIGN PRINCIPLES:
  1. Target ~40% error rate across models (enough wrong answers to measure CE)
  2. Questions should have ONE clearly correct answer (for deterministic scoring)
  3. Span multiple difficulty tiers: easy (anchor), medium (discriminate), hard (stress)
  4. Include "trap" questions with common wrong answers (tests monitoring sensitivity)
  5. Domains: math reasoning, counter-intuitive facts, disambiguation, multi-step logic

QUESTION CATEGORIES:
  A. Numerical traps — look easy but have common wrong answers
  B. Counter-intuitive facts — correct answer contradicts common belief
  C. Multi-step reasoning — requires 2-3 inferential steps
  D. Disambiguation — commonly confused entities/facts
  E. Edge cases — unusual exceptions to general rules
  F. Anchored easy — calibration anchors (models should get these right)

Usage:
    from mcc_ce_v2_questions import MCC_CE_V2_PROBLEMS, get_balanced_set
    problems = get_balanced_set(n=30)  # 30 balanced across difficulty
"""

import random

# Each entry: (question, correct_answer, domain, difficulty, category, common_wrong)
# common_wrong = the answer models are likely to give incorrectly (for trap analysis)

MCC_CE_V2_PROBLEMS = [
    # ── Category A: Numerical traps ──────────────────────────────────────────
    (
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?",
        "5", "math", "hard", "numerical_trap", "10"
    ),
    (
        "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?",
        "5", "math", "hard", "numerical_trap", "100"
    ),
    (
        "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days does it take to cover half the lake?",
        "47", "math", "hard", "numerical_trap", "24"
    ),
    (
        "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
        "9", "math", "medium", "numerical_trap", "8"
    ),
    (
        "How many times can you subtract 5 from 25?",
        "1", "math", "hard", "numerical_trap", "5"
    ),
    (
        "If you have a 7-minute hourglass and an 11-minute hourglass, what is the shortest time you can measure exactly?",
        "1", "math", "hard", "numerical_trap", "4"
    ),
    (
        "What is 0.1 + 0.2? Give the exact decimal answer.",
        "0.3", "math", "medium", "numerical_trap", "0.30000000000000004"
    ),
    (
        "A clock strikes 6 in 5 seconds. How many seconds does it take to strike 12?",
        "11", "math", "hard", "numerical_trap", "10"
    ),

    # ── Category B: Counter-intuitive facts ──────────────────────────────────
    (
        "Which country has the most time zones?",
        "France", "geography", "hard", "counter_intuitive", "Russia"
    ),
    (
        "What is the driest continent on Earth?",
        "Antarctica", "geography", "hard", "counter_intuitive", "Africa"
    ),
    (
        "Which planet in our solar system has the hottest surface temperature?",
        "Venus", "astronomy", "medium", "counter_intuitive", "Mercury"
    ),
    (
        "What is the longest river in Europe?",
        "Volga", "geography", "medium", "counter_intuitive", "Danube"
    ),
    (
        "Which ocean is the smallest?",
        "Arctic", "geography", "medium", "counter_intuitive", "Indian"
    ),
    (
        "What percentage of the Earth's water is fresh water: approximately 3%, 10%, or 25%?",
        "3%", "geography", "medium", "counter_intuitive", "10%"
    ),
    (
        "Which has more bones: an adult human or a baby?",
        "baby", "biology", "hard", "counter_intuitive", "adult"
    ),
    (
        "What is the national animal of Scotland?",
        "unicorn", "geography", "hard", "counter_intuitive", "lion"
    ),

    # ── Category C: Multi-step reasoning ─────────────────────────────────────
    (
        "If January 1st is a Monday, what day of the week is March 1st in a non-leap year?",
        "Thursday", "math", "hard", "multi_step", "Wednesday"
    ),
    (
        "A snail climbs 3 meters up a wall during the day but slides back 2 meters at night. How many days does it take to reach the top of a 10-meter wall?",
        "8", "math", "hard", "multi_step", "10"
    ),
    (
        "Three people check into a hotel room that costs $30. They each pay $10. The manager realizes the room only costs $25 and gives $5 to the bellboy to return. The bellboy keeps $2 and gives $1 back to each person. Now each person paid $9 (total $27) plus the $2 the bellboy kept = $29. Where is the missing dollar?",
        "There is no missing dollar; the $27 already includes the bellboy's $2", "math", "hard", "multi_step", "It disappeared"
    ),
    (
        "You have 12 balls, one of which is heavier. Using a balance scale, what is the minimum number of weighings needed to find the heavy ball?",
        "3", "math", "hard", "multi_step", "4"
    ),
    (
        "If you fold a standard piece of paper in half 42 times (assuming you could), approximately how thick would it be?",
        "About 440,000 km (past the Moon)", "math", "hard", "multi_step", "A few meters"
    ),
    (
        "Two trains are 100 km apart and moving toward each other, each at 50 km/h. A fly starts at one train and flies at 75 km/h back and forth between them. How far does the fly travel before the trains meet?",
        "75 km", "math", "hard", "multi_step", "100 km"
    ),

    # ── Category D: Disambiguation ───────────────────────────────────────────
    (
        "Who was the first person to set foot on the South Pole?",
        "Roald Amundsen", "history", "medium", "disambiguation", "Robert Scott"
    ),
    (
        "What is the largest desert in the world by area?",
        "Antarctic Desert", "geography", "hard", "disambiguation", "Sahara"
    ),
    (
        "Who invented the telephone?",
        "Alexander Graham Bell", "history", "easy", "disambiguation", "Elisha Gray"
    ),
    (
        "What is the hardest natural substance on Earth?",
        "Diamond", "science", "easy", "disambiguation", "Graphene"
    ),
    (
        "Which vitamin does the human body produce when exposed to sunlight?",
        "Vitamin D", "biology", "easy", "disambiguation", "Vitamin C"
    ),
    (
        "What was the first artificial satellite launched into space?",
        "Sputnik 1", "history", "easy", "disambiguation", "Explorer 1"
    ),

    # ── Category E: Edge cases ───────────────────────────────────────────────
    (
        "How many US presidents have been assassinated while in office?",
        "4", "history", "medium", "edge_case", "3"
    ),
    (
        "What is the only letter that does not appear in any US state name?",
        "Q", "geography", "hard", "edge_case", "X"
    ),
    (
        "How many hearts does an octopus have?",
        "3", "biology", "medium", "edge_case", "1"
    ),
    (
        "What is the only planet in our solar system not named after a god?",
        "Earth", "astronomy", "medium", "edge_case", "Pluto"
    ),
    (
        "In what year did the Berlin Wall fall?",
        "1989", "history", "easy", "edge_case", "1991"
    ),
    (
        "What is the chemical formula for table salt?",
        "NaCl", "chemistry", "easy", "edge_case", "Na2Cl"
    ),

    # ── Category F: Anchored easy (calibration) ──────────────────────────────
    (
        "What is 7 × 8?",
        "56", "math", "easy", "anchor", None
    ),
    (
        "What is the capital of Japan?",
        "Tokyo", "geography", "easy", "anchor", None
    ),
    (
        "How many continents are there?",
        "7", "geography", "easy", "anchor", None
    ),
    (
        "What is the chemical symbol for water?",
        "H2O", "chemistry", "easy", "anchor", None
    ),
    (
        "Who wrote Romeo and Juliet?",
        "Shakespeare", "literature", "easy", "anchor", None
    ),
    (
        "What is the square root of 64?",
        "8", "math", "easy", "anchor", None
    ),
    (
        "What color do you get when you mix red and blue?",
        "purple", "science", "easy", "anchor", None
    ),
    (
        "How many days are in a leap year?",
        "366", "science", "easy", "anchor", None
    ),
]


def get_problems_as_dicts() -> list[dict]:
    """Convert tuple format to dict format compatible with MCC pipeline."""
    return [
        {
            "question": q,
            "answer": a,
            "domain": d,
            "difficulty": diff,
            "category": cat,
            "common_wrong": cw,
        }
        for q, a, d, diff, cat, cw in MCC_CE_V2_PROBLEMS
    ]


def get_balanced_set(n: int = 30, seed: int = 42) -> list[dict]:
    """
    Return a balanced set of n problems:
      - ~20% easy anchors (calibration)
      - ~30% medium (discriminate mid-range)
      - ~50% hard (stress test, generate errors)

    This ratio targets ~40% error rate for mid-tier models.
    """
    rng = random.Random(seed)
    all_probs = get_problems_as_dicts()

    easy = [p for p in all_probs if p["difficulty"] == "easy"]
    medium = [p for p in all_probs if p["difficulty"] == "medium"]
    hard = [p for p in all_probs if p["difficulty"] == "hard"]

    n_easy = max(1, round(n * 0.2))
    n_medium = max(1, round(n * 0.3))
    n_hard = n - n_easy - n_medium

    # Sample with cycling if n > available
    def sample_cycle(pool, k):
        if k <= len(pool):
            return rng.sample(pool, k)
        result = pool * (k // len(pool) + 1)
        return rng.sample(result, k)

    selected = (
        sample_cycle(easy, n_easy) +
        sample_cycle(medium, n_medium) +
        sample_cycle(hard, n_hard)
    )
    rng.shuffle(selected)
    return selected


def get_trap_analysis_set(n: int = 20, seed: int = 42) -> list[dict]:
    """
    Return only trap questions (those with known common_wrong answers).
    These are the most diagnostic for MCC-CE: if a model gives the
    common_wrong answer, can it flag and correct it?
    """
    rng = random.Random(seed)
    all_probs = get_problems_as_dicts()
    traps = [p for p in all_probs if p["common_wrong"] is not None]
    if n <= len(traps):
        return rng.sample(traps, n)
    return rng.sample(traps * (n // len(traps) + 1), n)


# ── Summary statistics ───────────────────────────────────────────────────────

if __name__ == "__main__":
    all_probs = get_problems_as_dicts()
    print(f"Total problems: {len(all_probs)}")
    print(f"\nBy difficulty:")
    for d in ["easy", "medium", "hard"]:
        subset = [p for p in all_probs if p["difficulty"] == d]
        print(f"  {d}: {len(subset)}")
    print(f"\nBy category:")
    for cat in sorted(set(p["category"] for p in all_probs)):
        subset = [p for p in all_probs if p["category"] == cat]
        print(f"  {cat}: {len(subset)}")
    print(f"\nWith known common_wrong: {sum(1 for p in all_probs if p['common_wrong'] is not None)}")
    print(f"\nBalanced set (n=30):")
    balanced = get_balanced_set(30)
    for d in ["easy", "medium", "hard"]:
        print(f"  {d}: {sum(1 for p in balanced if p['difficulty'] == d)}")
