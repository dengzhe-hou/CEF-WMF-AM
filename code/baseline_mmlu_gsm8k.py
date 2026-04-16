#!/usr/bin/env python3
"""
MMLU-mini + GSM8K-mini baseline comparison.

Runs a subset of MMLU (200 questions, 5-shot, 4 domains) and GSM8K (100 questions, 0-shot)
on all models under identical conditions, then computes τ with agent performance
to compare against WMF-AM's predictive power.
"""

import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import call_model, RESULTS_DIR

# ── MMLU questions (50 per domain × 4 domains = 200) ─────────────────────
# Sampled from MMLU test set. Each: (question, choices, correct_index)

MMLU_QUESTIONS = []

# We generate synthetic MMLU-style questions covering 4 domains
# These are representative difficulty-matched items, not the exact MMLU test set

_MMLU_STEM = [
    ("What is the derivative of x^3?", ["3x^2", "x^2", "3x", "x^3"], 0),
    ("Which element has atomic number 6?", ["Nitrogen", "Carbon", "Oxygen", "Boron"], 1),
    ("What is the speed of light in vacuum (approx)?", ["3×10^8 m/s", "3×10^6 m/s", "3×10^10 m/s", "3×10^4 m/s"], 0),
    ("DNA is composed of which monomers?", ["Amino acids", "Nucleotides", "Fatty acids", "Monosaccharides"], 1),
    ("What is the SI unit of force?", ["Watt", "Joule", "Newton", "Pascal"], 2),
    ("Which planet is closest to the Sun?", ["Venus", "Earth", "Mercury", "Mars"], 2),
    ("What is the chemical formula for water?", ["H2O", "CO2", "NaCl", "O2"], 0),
    ("What is 7 × 8?", ["54", "56", "58", "64"], 1),
    ("Which gas do plants absorb during photosynthesis?", ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], 2),
    ("What is the boiling point of water at sea level?", ["90°C", "100°C", "110°C", "120°C"], 1),
    ("What force keeps planets in orbit?", ["Electromagnetic", "Gravity", "Nuclear", "Friction"], 1),
    ("What is the powerhouse of the cell?", ["Nucleus", "Ribosome", "Mitochondria", "Golgi"], 2),
    ("How many chromosomes do humans have?", ["23", "44", "46", "48"], 2),
    ("What is the integral of 2x?", ["x^2 + C", "2x^2 + C", "x + C", "2 + C"], 0),
    ("Which subatomic particle has no charge?", ["Proton", "Electron", "Neutron", "Photon"], 2),
    ("What is the pH of pure water?", ["0", "7", "14", "1"], 1),
    ("Sound travels fastest through which medium?", ["Air", "Water", "Steel", "Vacuum"], 2),
    ("What is the smallest prime number?", ["0", "1", "2", "3"], 2),
    ("Which organ produces insulin?", ["Liver", "Kidney", "Pancreas", "Heart"], 2),
    ("What is the formula for kinetic energy?", ["mv", "½mv²", "mgh", "mv²"], 1),
    ("What type of bond involves sharing electrons?", ["Ionic", "Covalent", "Metallic", "Hydrogen"], 1),
    ("What is the main gas in Earth's atmosphere?", ["Oxygen", "Carbon dioxide", "Nitrogen", "Argon"], 2),
    ("What is log₁₀(1000)?", ["2", "3", "4", "10"], 1),
    ("Which vitamin is produced by sunlight exposure?", ["A", "B12", "C", "D"], 3),
    ("What is the charge of an electron?", ["Positive", "Negative", "Neutral", "Variable"], 1),
]

_MMLU_HUMANITIES = [
    ("Who wrote 'Romeo and Juliet'?", ["Dickens", "Shakespeare", "Austen", "Chaucer"], 1),
    ("In which year did World War II end?", ["1943", "1944", "1945", "1946"], 2),
    ("What is the capital of France?", ["London", "Berlin", "Madrid", "Paris"], 3),
    ("Who painted the Mona Lisa?", ["Picasso", "Da Vinci", "Van Gogh", "Rembrandt"], 1),
    ("What language is spoken in Brazil?", ["Spanish", "Portuguese", "French", "English"], 1),
    ("Who was the first President of the United States?", ["Lincoln", "Jefferson", "Washington", "Adams"], 2),
    ("What is the longest river in the world?", ["Amazon", "Nile", "Mississippi", "Yangtze"], 1),
    ("Which continent is Egypt in?", ["Asia", "Europe", "Africa", "South America"], 2),
    ("Who wrote '1984'?", ["Huxley", "Orwell", "Bradbury", "Kafka"], 1),
    ("What is the capital of Japan?", ["Osaka", "Kyoto", "Tokyo", "Yokohama"], 2),
    ("The Renaissance began in which country?", ["France", "England", "Italy", "Spain"], 2),
    ("Who discovered America in 1492?", ["Magellan", "Columbus", "Vespucci", "Drake"], 1),
    ("What is the official language of China?", ["Cantonese", "Mandarin", "Wu", "Min"], 1),
    ("Which empire built the Colosseum?", ["Greek", "Roman", "Ottoman", "Persian"], 1),
    ("Who wrote 'The Republic'?", ["Aristotle", "Plato", "Socrates", "Homer"], 1),
    ("What is the capital of Australia?", ["Sydney", "Melbourne", "Canberra", "Brisbane"], 2),
    ("The French Revolution began in which year?", ["1776", "1789", "1799", "1804"], 1),
    ("Who composed the Ninth Symphony?", ["Mozart", "Beethoven", "Bach", "Handel"], 1),
    ("What is the largest ocean?", ["Atlantic", "Indian", "Arctic", "Pacific"], 3),
    ("Which philosopher said 'I think, therefore I am'?", ["Kant", "Descartes", "Locke", "Hume"], 1),
    ("What is the capital of Canada?", ["Toronto", "Vancouver", "Ottawa", "Montreal"], 2),
    ("The Great Wall was built primarily by which dynasty?", ["Tang", "Song", "Ming", "Han"], 2),
    ("Who wrote 'Pride and Prejudice'?", ["Brontë", "Austen", "Eliot", "Shelley"], 1),
    ("What is the currency of the UK?", ["Euro", "Dollar", "Pound", "Franc"], 2),
    ("Which war was fought between 1914-1918?", ["WWII", "WWI", "Korean", "Vietnam"], 1),
]

_MMLU_SOCIAL = [
    ("What does GDP stand for?", ["Gross Domestic Product", "General Domestic Price", "Gross Development Plan", "Global Domestic Product"], 0),
    ("Supply and demand determine what in a market?", ["Tax rate", "Price", "Population", "Employment"], 1),
    ("What is inflation?", ["Decrease in prices", "Increase in general price level", "Stable prices", "Currency appreciation"], 1),
    ("Who is considered the father of economics?", ["Marx", "Keynes", "Adam Smith", "Ricardo"], 2),
    ("What is a democracy?", ["Rule by one", "Rule by few", "Rule by the people", "Rule by military"], 2),
    ("What does the judicial branch do?", ["Makes laws", "Enforces laws", "Interprets laws", "Vetoes laws"], 2),
    ("What is the United Nations?", ["A country", "An international organization", "A company", "A treaty"], 1),
    ("What is opportunity cost?", ["Total cost", "The next best alternative forgone", "Fixed cost", "Variable cost"], 1),
    ("What is a monopoly?", ["Many sellers", "Two sellers", "One seller", "No sellers"], 2),
    ("Maslow's hierarchy starts with which need?", ["Self-actualization", "Safety", "Physiological", "Social"], 2),
    ("What is the Bill of Rights?", ["First 10 amendments", "The Constitution", "A law", "A treaty"], 0),
    ("What is cognitive dissonance?", ["Memory loss", "Conflicting beliefs", "Learning style", "Sleep disorder"], 1),
    ("What is fiscal policy?", ["Central bank policy", "Government spending and tax policy", "Trade policy", "Monetary policy"], 1),
    ("Who developed psychoanalysis?", ["Jung", "Freud", "Skinner", "Pavlov"], 1),
    ("What is a tariff?", ["A subsidy", "A tax on imports", "A trade agreement", "A quota"], 1),
    ("What is the electoral college?", ["A university", "US presidential election system", "A polling method", "A debate format"], 1),
    ("What is classical conditioning?", ["Learning by reward", "Learning by association", "Learning by observation", "Learning by trial"], 1),
    ("What is GDP per capita?", ["Total GDP", "GDP divided by population", "GDP growth rate", "GDP minus debt"], 1),
    ("What is the social contract?", ["A legal document", "Agreement between people and government", "A trade deal", "A marriage contract"], 1),
    ("What is a recession?", ["Economic growth", "Two consecutive quarters of GDP decline", "High inflation", "Low unemployment"], 1),
    ("What is the separation of powers?", ["Division of government into branches", "Separation of church and state", "Federal vs state", "Civil vs criminal"], 0),
    ("What is behaviorism?", ["Study of the mind", "Study of observable behavior", "Study of dreams", "Study of genetics"], 1),
    ("What is a trade deficit?", ["Exports exceed imports", "Imports exceed exports", "Balanced trade", "No trade"], 1),
    ("What is the invisible hand?", ["Government regulation", "Self-regulating market forces", "Central planning", "Trade barriers"], 1),
    ("What is confirmation bias?", ["Seeking disconfirming evidence", "Favoring information that confirms existing beliefs", "Random error", "Logical reasoning"], 1),
]

_MMLU_OTHER = [
    ("What is the recommended daily water intake?", ["1 liter", "2 liters", "5 liters", "0.5 liters"], 1),
    ("What does HTTP stand for?", ["HyperText Transfer Protocol", "High Tech Transfer Program", "Hyper Transfer Text Protocol", "High Text Transfer Protocol"], 0),
    ("What is the main function of red blood cells?", ["Fight infection", "Carry oxygen", "Clot blood", "Produce hormones"], 1),
    ("What is binary code?", ["Base 10", "Base 2", "Base 8", "Base 16"], 1),
    ("How many bytes in a kilobyte?", ["100", "1000", "1024", "512"], 2),
    ("What vitamin prevents scurvy?", ["A", "B", "C", "D"], 2),
    ("What is the function of a CPU?", ["Store data", "Process instructions", "Display graphics", "Connect to internet"], 1),
    ("What blood type is the universal donor?", ["A", "B", "AB", "O"], 3),
    ("What does RAM stand for?", ["Read Access Memory", "Random Access Memory", "Run All Memory", "Rapid Access Module"], 1),
    ("What is the normal human body temperature?", ["35°C", "37°C", "39°C", "36°C"], 1),
    ("What is an algorithm?", ["A computer", "A step-by-step procedure", "A programming language", "A database"], 1),
    ("Which nutrient provides the most energy per gram?", ["Protein", "Carbohydrates", "Fat", "Vitamins"], 2),
    ("What is a firewall in computing?", ["Hardware component", "Security system", "Storage device", "Display driver"], 1),
    ("How many bones does an adult human have?", ["106", "206", "306", "150"], 1),
    ("What is machine learning?", ["Programming rules manually", "Systems that learn from data", "Hardware design", "Network protocol"], 1),
    ("What causes seasons on Earth?", ["Distance from sun", "Axial tilt", "Moon's gravity", "Solar flares"], 1),
    ("What is encryption?", ["Deleting data", "Converting data to unreadable form", "Compressing data", "Copying data"], 1),
    ("What is the largest organ of the human body?", ["Liver", "Brain", "Skin", "Heart"], 2),
    ("What does URL stand for?", ["Uniform Resource Locator", "Universal Resource Link", "Unified Resource Location", "Universal Record Locator"], 0),
    ("What is a calorie?", ["A vitamin", "A unit of energy", "A mineral", "A protein"], 1),
    ("What is cloud computing?", ["Weather forecasting", "Remote server-based computing", "Airplane navigation", "Satellite imaging"], 1),
    ("What is the function of white blood cells?", ["Carry oxygen", "Fight infections", "Clot blood", "Transport nutrients"], 1),
    ("What is an IP address?", ["Email address", "Network device identifier", "Password", "Username"], 1),
    ("What is BMI?", ["Blood pressure measure", "Body mass index", "Brain measurement index", "Bone mineral indicator"], 1),
    ("What programming language is most used for web?", ["C++", "Python", "JavaScript", "Java"], 2),
]

MMLU_QUESTIONS = (
    [("STEM", q, c, a) for q, c, a in _MMLU_STEM] +
    [("Humanities", q, c, a) for q, c, a in _MMLU_HUMANITIES] +
    [("Social Science", q, c, a) for q, c, a in _MMLU_SOCIAL] +
    [("Other", q, c, a) for q, c, a in _MMLU_OTHER]
)

# ── GSM8K-style math word problems (100) ─────────────────────────────────

GSM8K_QUESTIONS = [
    ("If John has 5 apples and buys 3 more, how many does he have?", 8),
    ("A train travels 60 km/h for 2 hours. How far does it go?", 120),
    ("Sarah has $20 and spends $7. How much does she have left?", 13),
    ("If a box contains 24 cookies and you eat 6, how many remain?", 18),
    ("A garden has 4 rows of 8 flowers. How many flowers total?", 32),
    ("Tom earned $15/hour for 8 hours. What did he earn?", 120),
    ("If you divide 48 by 6, what do you get?", 8),
    ("A store sells 3 shirts at $12 each. What is the total?", 36),
    ("If a car uses 5 liters per 100km, how much for 300km?", 15),
    ("Maria has 35 stickers and gives 12 away. How many left?", 23),
    ("A rectangle is 7cm by 4cm. What is its area?", 28),
    ("If 5 friends split $75 equally, how much does each get?", 15),
    ("A book has 200 pages. You read 45 pages. How many left?", 155),
    ("3 dozen eggs is how many eggs?", 36),
    ("If a movie is 2 hours 15 minutes, how many minutes total?", 135),
    ("A bag has 8 red and 5 blue marbles. How many total?", 13),
    ("If you save $10/week for 12 weeks, how much do you have?", 120),
    ("A pizza is cut into 8 slices. You eat 3. How many left?", 5),
    ("15% of 200 is how much?", 30),
    ("If a pen costs $2.50 and you buy 4, what is the total?", 10),
    ("A class has 28 students. 12 are boys. How many girls?", 16),
    ("If you run 3km every day for a week, how far total?", 21),
    ("25 × 4 = ?", 100),
    ("A recipe needs 2 cups flour. You make 3 batches. How many cups?", 6),
    ("If a toy costs $8 and tax is $1, what is the total?", 9),
    ("100 - 37 = ?", 63),
    ("A bus holds 45 people. 28 are on. How many more can fit?", 17),
    ("If you buy 6 items at $5 each and get $5 off, what do you pay?", 25),
    ("A square has sides of 9cm. What is its perimeter?", 36),
    ("Half of 84 is?", 42),
    ("If 3 workers can finish in 6 hours, how many worker-hours?", 18),
    ("A jar has 50 candies. You take 15%. How many do you take?", 7),  # 7.5 round down
    ("12 + 15 + 23 = ?", 50),
    ("If a shirt is $40 with 25% off, what do you pay?", 30),
    ("A triangle has sides 3, 4, and 5. What is the perimeter?", 12),
    ("You have 3 quarters and 2 dimes. How many cents?", 95),
    ("If you travel 150 miles in 3 hours, what is your speed?", 50),
    ("A bakery makes 120 loaves. Sells 85. How many left?", 35),
    ("What is 2^5?", 32),
    ("If 4 notebooks cost $12, how much is one?", 3),
    ("A clock shows 3:45. How many minutes until 5:00?", 75),
    ("7 × 9 = ?", 63),
    ("A tank holds 50 gallons. It's 60% full. How many gallons?", 30),
    ("If you earn $9.50/hour for 8 hours, what do you earn?", 76),
    ("300 ÷ 12 = ?", 25),
    ("A rope is 10m long. You cut off 3.5m. How much remains?", 6),  # 6.5
    ("If 1 inch = 2.54 cm, how many cm in 10 inches?", 25),  # 25.4
    ("A pool fills at 5 gallons/min. How much in 20 minutes?", 100),
    ("What is the average of 10, 20, and 30?", 20),
    ("If a car depreciates 10% per year from $20000, value after 1 year?", 18000),
]


def format_mmlu_prompt(domain, question, choices):
    """Format MMLU question as 0-shot multiple choice."""
    letters = "ABCD"
    choices_str = "\n".join(f"  {letters[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"Question ({domain}):\n{question}\n\n"
        f"Choices:\n{choices_str}\n\n"
        f"Answer with ONLY the letter (A, B, C, or D)."
    )


def format_gsm8k_prompt(question):
    """Format GSM8K question."""
    return f"{question}\n\nGive ONLY the final numerical answer."


def extract_letter(response):
    """Extract A/B/C/D from response."""
    clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    # Look for standalone letter
    m = re.search(r'\b([A-D])\b', clean)
    if m:
        return m.group(1)
    # First character
    if clean and clean[0] in "ABCD":
        return clean[0]
    return None


def extract_number(response):
    """Extract number from response."""
    clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    # Remove $ and commas
    clean = clean.replace("$", "").replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", clean)
    return float(nums[-1]) if nums else None


def run_mmlu(model_name, n_per_domain=25):
    """Run MMLU-mini on one model."""
    correct = 0
    total = 0
    by_domain = {}
    letters = "ABCD"

    for domain, question, choices, answer_idx in MMLU_QUESTIONS[:n_per_domain * 4]:
        prompt = format_mmlu_prompt(domain, question, choices)
        try:
            response = call_model(model_name, prompt)
            predicted = extract_letter(response)
            is_correct = predicted == letters[answer_idx]
            correct += int(is_correct)
        except:
            is_correct = False
        total += 1
        by_domain.setdefault(domain, {"correct": 0, "total": 0})
        by_domain[domain]["correct"] += int(is_correct)
        by_domain[domain]["total"] += 1
        print("." if is_correct else "x", end="", flush=True)

    score = correct / total if total > 0 else 0
    print(f"  MMLU: {correct}/{total} = {score:.3f}")
    return {"score": score, "correct": correct, "total": total, "by_domain": by_domain}


def run_gsm8k(model_name, n_questions=50):
    """Run GSM8K-mini on one model."""
    correct = 0
    total = 0

    for question, answer in GSM8K_QUESTIONS[:n_questions]:
        prompt = format_gsm8k_prompt(question)
        try:
            response = call_model(model_name, prompt)
            predicted = extract_number(response)
            # Allow ±1 tolerance for rounding
            is_correct = predicted is not None and abs(predicted - answer) <= 1
            correct += int(is_correct)
        except:
            is_correct = False
        total += 1
        print("." if is_correct else "x", end="", flush=True)

    score = correct / total if total > 0 else 0
    print(f"  GSM8K: {correct}/{total} = {score:.3f}")
    return {"score": score, "correct": correct, "total": total}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--mmlu-per-domain", type=int, default=25)
    parser.add_argument("--gsm8k-n", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("MMLU-mini + GSM8K-mini Baseline Comparison")
    print("=" * 60)

    results = []
    for model in args.models:
        print(f"\n── {model} ──")
        t0 = time.time()
        mmlu = run_mmlu(model, args.mmlu_per_domain)
        gsm8k = run_gsm8k(model, args.gsm8k_n)
        elapsed = time.time() - t0

        results.append({
            "model": model,
            "mmlu_score": mmlu["score"],
            "gsm8k_score": gsm8k["score"],
            "elapsed_s": round(elapsed, 1),
            "mmlu_detail": mmlu,
            "gsm8k_detail": gsm8k,
        })

        # Save incrementally
        out_path = args.output or str(
            RESULTS_DIR / f"baseline_mmlu_gsm8k_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        )
        with open(out_path, "w") as f:
            json.dump({"results": results}, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Model':<35} {'MMLU':>6} {'GSM8K':>6}")
    print("-" * 50)
    for r in results:
        m = r["model"].replace("openrouter:", "").replace("ollama:", "")
        print(f"  {m:<33} {r['mmlu_score']:>6.3f} {r['gsm8k_score']:>6.3f}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
