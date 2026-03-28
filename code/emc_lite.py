"""
EMC-lite: Episodic Memory Coherence — Fault Detection & Recovery Probes

Three trial families (12 trials each, 36 total per model):
  1. Stale-update:       One episode with a late correction; query post-correction value.
  2. Cross-episode lure: Three similar episodes with overlapping slots + correction.
  3. Source-conflict:    Two sources disagree; authority implied via indirect cues; query current fact.

Each trial has a matched NO-FAULT control twin (same structure, no contradiction).

Scoring is fully deterministic via structured JSON output with fact IDs.

Usage:
    python emc_lite.py --pilot                          # 3 models × 2 conditions × 18 items
    python emc_lite.py --all-ollama                     # all 7 models × 2 conditions × 36 items
    python emc_lite.py --models ollama:qwen2.5:7b ollama:deepseek-r1:14b ollama:llama3.1:8b
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, RESULTS_DIR, call_model

RANDOM_SEED = 42

# ── Filler turns (irrelevant interference between correction and query) ────
FILLER_TURNS = [
    "Reminder: Building maintenance is scheduled for this Saturday 8 AM–12 PM. Please move vehicles from the east lot.",
    "Weather advisory: Partly cloudy skies expected through Thursday, high of 68°F. No travel disruptions anticipated.",
    "Note: The office supply order for Q2 has been submitted. Delivery expected within 5 business days.",
    "FYI: The shared kitchen refrigerator will be cleaned out this Friday at 5 PM. Please label all items.",
    "Scheduling update: The all-hands meeting has been moved to the second Tuesday of each month going forward.",
    "Parking notice: Visitor parking passes are now available at the front desk. Please collect before 9 AM.",
    "IT advisory: System patches will be applied tonight between 11 PM and 2 AM. Brief outages may occur.",
    "Cafeteria menu change: Thursday's lunch special has been updated to grilled salmon with seasonal vegetables.",
    "Reminder: Timesheets for the current pay period are due by end of day Friday.",
    "Facilities note: The third-floor water fountain is temporarily out of service. Use second-floor alternatives.",
    "HR update: Open enrollment for benefits packages closes at the end of this month.",
    "Security notice: Please ensure all visitors sign in at reception and wear visitor badges at all times.",
]


def _sample_fillers(rng: random.Random, n: int = 3) -> list[str]:
    """Return n random filler sentences for interference injection."""
    return rng.sample(FILLER_TURNS, min(n, len(FILLER_TURNS)))


# ── Stimulus banks ──────────────────────────────────────────────────────────

# Domains for stale-update trials
STALE_UPDATE_DOMAINS = [
    {
        "context": "project management",
        "entities": {
            "S1": ("deadline", "March 15"),
            "S2": ("budget", "$50,000"),
            "S3": ("team_lead", "Alice"),
            "S4": ("status", "on track"),
        },
        "correction": ("S2", "budget", "$65,000"),
        "query_id": "S2",
    },
    {
        "context": "flight booking",
        "entities": {
            "S1": ("departure", "10:30 AM"),
            "S2": ("gate", "B12"),
            "S3": ("seat", "14A"),
            "S4": ("terminal", "Terminal 2"),
        },
        "correction": ("S2", "gate", "C7"),
        "query_id": "S2",
    },
    {
        "context": "medical appointment",
        "entities": {
            "S1": ("doctor", "Dr. Chen"),
            "S2": ("date", "April 3"),
            "S3": ("time", "2:00 PM"),
            "S4": ("location", "Room 204"),
        },
        "correction": ("S3", "time", "3:30 PM"),
        "query_id": "S3",
    },
    {
        "context": "restaurant reservation",
        "entities": {
            "S1": ("restaurant", "La Piazza"),
            "S2": ("time", "7:00 PM"),
            "S3": ("party_size", "4 people"),
            "S4": ("special_request", "window table"),
        },
        "correction": ("S2", "time", "8:30 PM"),
        "query_id": "S2",
    },
    {
        "context": "server configuration",
        "entities": {
            "S1": ("hostname", "prod-web-03"),
            "S2": ("port", "8080"),
            "S3": ("memory", "16 GB"),
            "S4": ("region", "us-east-1"),
        },
        "correction": ("S2", "port", "9090"),
        "query_id": "S2",
    },
    {
        "context": "rental agreement",
        "entities": {
            "S1": ("monthly_rent", "$1,200"),
            "S2": ("move_in_date", "June 1"),
            "S3": ("deposit", "$2,400"),
            "S4": ("lease_term", "12 months"),
        },
        "correction": ("S1", "monthly_rent", "$1,350"),
        "query_id": "S1",
    },
    {
        "context": "course registration",
        "entities": {
            "S1": ("course", "CS 301"),
            "S2": ("instructor", "Prof. Kim"),
            "S3": ("schedule", "MWF 10:00 AM"),
            "S4": ("room", "Hall 205"),
        },
        "correction": ("S4", "room", "Lab 112"),
        "query_id": "S4",
    },
    {
        "context": "shipping order",
        "entities": {
            "S1": ("tracking_number", "TRK-98712"),
            "S2": ("delivery_date", "March 20"),
            "S3": ("carrier", "FedEx"),
            "S4": ("weight", "5.2 kg"),
        },
        "correction": ("S2", "delivery_date", "March 25"),
        "query_id": "S2",
    },
    {
        "context": "event planning",
        "entities": {
            "S1": ("venue", "Grand Hall"),
            "S2": ("date", "May 10"),
            "S3": ("capacity", "200 guests"),
            "S4": ("catering", "Sunrise Foods"),
        },
        "correction": ("S1", "venue", "Riverside Pavilion"),
        "query_id": "S1",
    },
    {
        "context": "job offer",
        "entities": {
            "S1": ("salary", "$95,000"),
            "S2": ("start_date", "July 1"),
            "S3": ("title", "Senior Engineer"),
            "S4": ("team", "Infrastructure"),
        },
        "correction": ("S1", "salary", "$102,000"),
        "query_id": "S1",
    },
    {
        "context": "vehicle maintenance",
        "entities": {
            "S1": ("mileage", "45,000 miles"),
            "S2": ("next_service", "October 15"),
            "S3": ("tire_pressure", "35 PSI"),
            "S4": ("oil_type", "5W-30"),
        },
        "correction": ("S2", "next_service", "September 28"),
        "query_id": "S2",
    },
    {
        "context": "budget report",
        "entities": {
            "S1": ("total_revenue", "$340,000"),
            "S2": ("expenses", "$280,000"),
            "S3": ("net_profit", "$60,000"),
            "S4": ("quarter", "Q3"),
        },
        "correction": ("S2", "expenses", "$295,000"),
        "query_id": "S2",
    },
]

# Domains for cross-episode lure trials
CROSS_EPISODE_DOMAINS = [
    {
        "context": "meeting scheduling",
        "episode_a": {
            "S1": ("client", "Acme Corp"),
            "S2": ("time", "2:00 PM"),
            "S3": ("topic", "Q3 review"),
        },
        "episode_b": {
            "S4": ("client", "Beta Inc"),
            "S5": ("time", "3:30 PM"),
            "S6": ("topic", "Q3 forecast"),
        },
        "episode_c": {
            "S7": ("client", "Gamma LLC"),
            "S8": ("time", "2:15 PM"),
            "S9": ("topic", "Q3 planning"),
        },
        "correction_episode": "B",
        "correction": ("S5", "time", "4:00 PM"),
        "query_episode": "B",
        "query_id": "S5",
    },
    {
        "context": "patient records",
        "episode_a": {
            "S1": ("patient", "John Smith"),
            "S2": ("medication", "Lisinopril 10mg"),
            "S3": ("appointment", "Monday 9 AM"),
        },
        "episode_b": {
            "S4": ("patient", "Jane Smith"),
            "S5": ("medication", "Lisinopril 20mg"),
            "S6": ("appointment", "Monday 10 AM"),
        },
        "episode_c": {
            "S7": ("patient", "James Smith"),
            "S8": ("medication", "Lisinopril 15mg"),
            "S9": ("appointment", "Monday 11 AM"),
        },
        "correction_episode": "A",
        "correction": ("S2", "medication", "Lisinopril 5mg"),
        "query_episode": "A",
        "query_id": "S2",
    },
    {
        "context": "warehouse inventory",
        "episode_a": {
            "S1": ("product", "Widget-A"),
            "S2": ("quantity", "500 units"),
            "S3": ("location", "Shelf 12"),
        },
        "episode_b": {
            "S4": ("product", "Widget-B"),
            "S5": ("quantity", "480 units"),
            "S6": ("location", "Shelf 14"),
        },
        "episode_c": {
            "S7": ("product", "Widget-C"),
            "S8": ("quantity", "510 units"),
            "S9": ("location", "Shelf 13"),
        },
        "correction_episode": "A",
        "correction": ("S2", "quantity", "320 units"),
        "query_episode": "A",
        "query_id": "S2",
    },
    {
        "context": "travel itinerary",
        "episode_a": {
            "S1": ("destination", "Paris"),
            "S2": ("hotel", "Hotel Lumiere"),
            "S3": ("check_in", "March 5"),
        },
        "episode_b": {
            "S4": ("destination", "London"),
            "S5": ("hotel", "The Crown Hotel"),
            "S6": ("check_in", "March 8"),
        },
        "episode_c": {
            "S7": ("destination", "Berlin"),
            "S8": ("hotel", "Hotel Adler"),
            "S9": ("check_in", "March 6"),
        },
        "correction_episode": "B",
        "correction": ("S5", "hotel", "Kensington Arms"),
        "query_episode": "B",
        "query_id": "S5",
    },
    {
        "context": "student grades",
        "episode_a": {
            "S1": ("student", "Emily"),
            "S2": ("grade", "B+"),
            "S3": ("course", "History 201"),
        },
        "episode_b": {
            "S4": ("student", "Eric"),
            "S5": ("grade", "B"),
            "S6": ("course", "History 202"),
        },
        "episode_c": {
            "S7": ("student", "Elena"),
            "S8": ("grade", "B-"),
            "S9": ("course", "History 203"),
        },
        "correction_episode": "B",
        "correction": ("S5", "grade", "A-"),
        "query_episode": "B",
        "query_id": "S5",
    },
    {
        "context": "software deployment",
        "episode_a": {
            "S1": ("service", "auth-service"),
            "S2": ("version", "v2.3.1"),
            "S3": ("status", "deployed"),
        },
        "episode_b": {
            "S4": ("service", "api-gateway"),
            "S5": ("version", "v2.3.0"),
            "S6": ("status", "deployed"),
        },
        "episode_c": {
            "S7": ("service", "auth-proxy"),
            "S8": ("version", "v2.3.2"),
            "S9": ("status", "deployed"),
        },
        "correction_episode": "B",
        "correction": ("S5", "version", "v2.4.0"),
        "query_episode": "B",
        "query_id": "S5",
    },
    {
        "context": "insurance claims",
        "episode_a": {
            "S1": ("claim_id", "CLM-4401"),
            "S2": ("amount", "$12,500"),
            "S3": ("status", "under review"),
        },
        "episode_b": {
            "S4": ("claim_id", "CLM-4402"),
            "S5": ("amount", "$11,800"),
            "S6": ("status", "under review"),
        },
        "episode_c": {
            "S7": ("claim_id", "CLM-4403"),
            "S8": ("amount", "$12,100"),
            "S9": ("status", "under review"),
        },
        "correction_episode": "A",
        "correction": ("S2", "amount", "$14,200"),
        "query_episode": "A",
        "query_id": "S2",
    },
    {
        "context": "recipe database",
        "episode_a": {
            "S1": ("dish", "pasta primavera"),
            "S2": ("cook_time", "25 minutes"),
            "S3": ("servings", "4"),
        },
        "episode_b": {
            "S4": ("dish", "pasta carbonara"),
            "S5": ("cook_time", "20 minutes"),
            "S6": ("servings", "4"),
        },
        "episode_c": {
            "S7": ("dish", "pasta arrabiata"),
            "S8": ("cook_time", "22 minutes"),
            "S9": ("servings", "4"),
        },
        "correction_episode": "A",
        "correction": ("S2", "cook_time", "35 minutes"),
        "query_episode": "A",
        "query_id": "S2",
    },
    {
        "context": "real estate listings",
        "episode_a": {
            "S1": ("address", "42 Oak Street"),
            "S2": ("price", "$450,000"),
            "S3": ("bedrooms", "3"),
        },
        "episode_b": {
            "S4": ("address", "48 Oak Street"),
            "S5": ("price", "$475,000"),
            "S6": ("bedrooms", "3"),
        },
        "episode_c": {
            "S7": ("address", "44 Oak Street"),
            "S8": ("price", "$460,000"),
            "S9": ("bedrooms", "3"),
        },
        "correction_episode": "B",
        "correction": ("S5", "price", "$520,000"),
        "query_episode": "B",
        "query_id": "S5",
    },
    {
        "context": "subscription plans",
        "episode_a": {
            "S1": ("plan", "Basic"),
            "S2": ("price", "$9.99/month"),
            "S3": ("storage", "50 GB"),
        },
        "episode_b": {
            "S4": ("plan", "Premium"),
            "S5": ("price", "$19.99/month"),
            "S6": ("storage", "200 GB"),
        },
        "episode_c": {
            "S7": ("plan", "Standard"),
            "S8": ("price", "$14.99/month"),
            "S9": ("storage", "100 GB"),
        },
        "correction_episode": "B",
        "correction": ("S5", "price", "$24.99/month"),
        "query_episode": "B",
        "query_id": "S5",
    },
    {
        "context": "vendor contracts",
        "episode_a": {
            "S1": ("vendor", "CloudServe"),
            "S2": ("contract_value", "$180,000"),
            "S3": ("term", "2 years"),
        },
        "episode_b": {
            "S4": ("vendor", "DataFlow"),
            "S5": ("contract_value", "$165,000"),
            "S6": ("term", "2 years"),
        },
        "episode_c": {
            "S7": ("vendor", "CloudSync"),
            "S8": ("contract_value", "$175,000"),
            "S9": ("term", "2 years"),
        },
        "correction_episode": "A",
        "correction": ("S2", "contract_value", "$210,000"),
        "query_episode": "A",
        "query_id": "S2",
    },
    {
        "context": "lab experiments",
        "episode_a": {
            "S1": ("experiment", "Trial-A"),
            "S2": ("temperature", "72°C"),
            "S3": ("duration", "45 minutes"),
        },
        "episode_b": {
            "S4": ("experiment", "Trial-B"),
            "S5": ("temperature", "75°C"),
            "S6": ("duration", "50 minutes"),
        },
        "episode_c": {
            "S7": ("experiment", "Trial-C"),
            "S8": ("temperature", "73°C"),
            "S9": ("duration", "48 minutes"),
        },
        "correction_episode": "A",
        "correction": ("S2", "temperature", "68°C"),
        "query_episode": "A",
        "query_id": "S2",
    },
]

# Domains for source-conflict trials
SOURCE_CONFLICT_DOMAINS = [
    {
        "context": "product pricing",
        "source_a": {"name": "Sales Team email", "fact_id": "S1", "key": "unit price", "value": "$45"},
        "source_b": {"name": "Pricing Database", "fact_id": "S2", "key": "unit price", "value": "$52"},
        "authoritative": "B",
        "auth_label": "Company ERP system — automated feed", "stale_label": "Forwarded email thread",
        "auth_date": "March 10, 2026", "stale_date": "February 2, 2025",
        "query_key": "unit price",
    },
    {
        "context": "meeting room",
        "source_a": {"name": "Calendar invite", "fact_id": "S1", "key": "room", "value": "Conference A"},
        "source_b": {"name": "Facilities update", "fact_id": "S2", "key": "room", "value": "Conference D"},
        "authoritative": "A",  # counterbalanced
        "auth_label": "Shared team calendar — synced", "stale_label": "Bulletin board posting",
        "auth_date": "March 14, 2026", "stale_date": "November 18, 2024",
        "query_key": "room",
    },
    {
        "context": "delivery estimate",
        "source_a": {"name": "Original order confirmation", "fact_id": "S1", "key": "delivery date", "value": "March 18"},
        "source_b": {"name": "Updated shipping notification", "fact_id": "S2", "key": "delivery date", "value": "March 22"},
        "authoritative": "A",  # counterbalanced
        "auth_label": "Carrier tracking system", "stale_label": "Customer service chat transcript",
        "auth_date": "March 13, 2026", "stale_date": "March 1, 2025",
        "query_key": "delivery date",
    },
    {
        "context": "employee directory",
        "source_a": {"name": "HR database", "fact_id": "S1", "key": "office number", "value": "Room 305"},
        "source_b": {"name": "Recent move notification", "fact_id": "S2", "key": "office number", "value": "Room 412"},
        "authoritative": "B",
        "auth_label": "Facilities management system", "stale_label": "Employee self-reported profile",
        "auth_date": "March 11, 2026", "stale_date": "August 22, 2024",
        "query_key": "office number",
    },
    {
        "context": "software version",
        "source_a": {"name": "README file", "fact_id": "S1", "key": "version", "value": "3.1.0"},
        "source_b": {"name": "Release notes", "fact_id": "S2", "key": "version", "value": "3.2.1"},
        "authoritative": "A",  # counterbalanced
        "auth_label": "CI/CD pipeline artifact", "stale_label": "Developer's local changelog",
        "auth_date": "March 12, 2026", "stale_date": "December 5, 2024",
        "query_key": "version",
    },
    {
        "context": "API rate limit",
        "source_a": {"name": "Documentation page", "fact_id": "S1", "key": "rate limit", "value": "100 requests/min"},
        "source_b": {"name": "API status dashboard", "fact_id": "S2", "key": "rate limit", "value": "150 requests/min"},
        "authoritative": "B",
        "auth_label": "Live infrastructure dashboard", "stale_label": "Cached documentation page",
        "auth_date": "March 14, 2026", "stale_date": "October 30, 2024",
        "query_key": "rate limit",
    },
    {
        "context": "tax filing",
        "source_a": {"name": "Accountant's draft", "fact_id": "S1", "key": "total deductions", "value": "$18,500"},
        "source_b": {"name": "Revised accountant letter", "fact_id": "S2", "key": "total deductions", "value": "$21,300"},
        "authoritative": "A",  # counterbalanced
        "auth_label": "Signed CPA filing document", "stale_label": "Preliminary worksheet",
        "auth_date": "March 8, 2026", "stale_date": "January 15, 2025",
        "query_key": "total deductions",
    },
    {
        "context": "course syllabus",
        "source_a": {"name": "Original syllabus", "fact_id": "S1", "key": "final exam date", "value": "December 12"},
        "source_b": {"name": "Professor's announcement", "fact_id": "S2", "key": "final exam date", "value": "December 15"},
        "authoritative": "B",
        "auth_label": "University registrar portal", "stale_label": "Student forum post",
        "auth_date": "March 10, 2026", "stale_date": "September 3, 2024",
        "query_key": "final exam date",
    },
    {
        "context": "flight information",
        "source_a": {"name": "Booking confirmation", "fact_id": "S1", "key": "departure time", "value": "6:15 AM"},
        "source_b": {"name": "Airline app notification", "fact_id": "S2", "key": "departure time", "value": "7:45 AM"},
        "authoritative": "A",  # counterbalanced
        "auth_label": "Airport departure board feed", "stale_label": "Travel agent itinerary",
        "auth_date": "March 15, 2026", "stale_date": "February 20, 2025",
        "query_key": "departure time",
    },
    {
        "context": "contract terms",
        "source_a": {"name": "Draft contract", "fact_id": "S1", "key": "payment terms", "value": "Net 30"},
        "source_b": {"name": "Signed final contract", "fact_id": "S2", "key": "payment terms", "value": "Net 45"},
        "authoritative": "B",
        "auth_label": "Legal department executed copy", "stale_label": "Internal discussion memo",
        "auth_date": "March 9, 2026", "stale_date": "November 12, 2024",
        "query_key": "payment terms",
    },
    {
        "context": "server maintenance",
        "source_a": {"name": "Initial notice", "fact_id": "S1", "key": "maintenance window", "value": "Saturday 2-4 AM"},
        "source_b": {"name": "Updated notice from ops team", "fact_id": "S2", "key": "maintenance window", "value": "Sunday 1-3 AM"},
        "authoritative": "A",  # counterbalanced
        "auth_label": "Operations runbook — version controlled", "stale_label": "Slack channel message",
        "auth_date": "March 13, 2026", "stale_date": "January 28, 2025",
        "query_key": "maintenance window",
    },
    {
        "context": "warehouse stock",
        "source_a": {"name": "Morning inventory count", "fact_id": "S1", "key": "stock level", "value": "1,240 units"},
        "source_b": {"name": "Afternoon recount", "fact_id": "S2", "key": "stock level", "value": "1,185 units"},
        "authoritative": "B",
        "auth_label": "Warehouse management system scan", "stale_label": "Shift supervisor's handwritten log",
        "auth_date": "March 14, 2026", "stale_date": "March 1, 2025",
        "query_key": "stock level",
    },
]


# ── Prompt builders ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise information tracking assistant. "
    "Always respond with ONLY valid JSON in the exact format requested. "
    "No explanations, no markdown, no extra text."
)


def build_stale_update_prompt(domain: dict, inject_fault: bool = True) -> tuple[str, dict]:
    """Build a stale-update trial. Returns (prompt, expected_answer)."""
    ents = domain["entities"]
    corr_id, corr_key, corr_new_val = domain["correction"]
    query_id = domain["query_id"]

    # Build initial facts
    lines = [f"Context: {domain['context']}. Here are the current facts:"]
    for fid, (key, val) in ents.items():
        lines.append(f"  [{fid}] {key}: {val}")

    # Use a local RNG seeded from domain content for reproducible filler selection
    _filler_rng = random.Random(domain["context"])

    if inject_fault:
        # Add correction (reduced salience — no bold/asterisks)
        old_val = ents[corr_id][1]
        lines.append(f"\nNote: An update has been received.")
        lines.append(f"The {corr_key} ([{corr_id}]) has been changed "
                      f"from {old_val} to {corr_new_val}.")
        expected_answer = corr_new_val
        stale_ids = [corr_id]
    else:
        # No-fault control: restate the same value as "confirmed"
        old_val = ents[corr_id][1]
        lines.append(f"\nNote: A verification has been completed.")
        lines.append(f"The {corr_key} ([{corr_id}]) has been "
                      f"verified as {old_val}.")
        expected_answer = ents[query_id][1]
        stale_ids = []

    # Inject filler turns between correction/verification and query
    fillers = _sample_fillers(_filler_rng, n=3)
    for filler in fillers:
        lines.append(f"\n{filler}")
    lines.append("")

    query_key = ents[query_id][0]
    lines.append(f"Based on all the information above, what is the current {query_key}?")
    lines.append(f"Which fact IDs (if any) contained stale/outdated information before the update?")
    lines.append("")
    lines.append('Respond with ONLY this JSON (no other text):')
    lines.append('{"stale_ids": ["<id>", ...], "final_answer": "<value>"}')

    prompt = "\n".join(lines)
    expected = {
        "stale_ids": stale_ids,
        "final_answer": expected_answer,
    }
    return prompt, expected


def build_cross_episode_prompt(domain: dict, inject_fault: bool = True) -> tuple[str, dict]:
    """Build a cross-episode lure trial."""
    has_episode_c = "episode_c" in domain
    n_episodes = 3 if has_episode_c else 2
    lines = [f"Context: {domain['context']}. You will see {n_episodes} similar episodes."]

    # Episode A
    lines.append("\n--- Episode A ---")
    for fid, (key, val) in domain["episode_a"].items():
        lines.append(f"  [{fid}] {key}: {val}")

    # Episode B
    lines.append("\n--- Episode B ---")
    for fid, (key, val) in domain["episode_b"].items():
        lines.append(f"  [{fid}] {key}: {val}")

    # Episode C (interference episode)
    if has_episode_c:
        lines.append("\n--- Episode C ---")
        for fid, (key, val) in domain["episode_c"].items():
            lines.append(f"  [{fid}] {key}: {val}")

    corr_id, corr_key, corr_new_val = domain["correction"]
    corr_ep = domain["correction_episode"]
    query_ep = domain["query_episode"]
    query_id = domain["query_id"]

    _filler_rng = random.Random(domain["context"] + "_cross")

    if inject_fault:
        all_ents = {**domain["episode_a"], **domain["episode_b"],
                    **domain.get("episode_c", {})}
        old_val = all_ents[corr_id][1]
        lines.append(f"\nNote: An update has been received for Episode {corr_ep}.")
        lines.append(f"The {corr_key} ([{corr_id}]) has changed "
                      f"from {old_val} to {corr_new_val}.")
        expected_answer = corr_new_val
        stale_ids = [corr_id]
    else:
        all_ents = {**domain["episode_a"], **domain["episode_b"],
                    **domain.get("episode_c", {})}
        old_val = all_ents[corr_id][1]
        lines.append(f"\nNote: A verification for Episode {corr_ep} is complete.")
        lines.append(f"The {corr_key} ([{corr_id}]) is confirmed as {old_val}.")
        expected_answer = all_ents[query_id][1]
        stale_ids = []

    # Inject filler turns between correction/verification and query
    fillers = _sample_fillers(_filler_rng, n=3)
    for filler in fillers:
        lines.append(f"\n{filler}")
    lines.append("")

    query_key = ({**domain["episode_a"], **domain["episode_b"],
                  **domain.get("episode_c", {})})[query_id][0]
    lines.append(f"For Episode {query_ep}: What is the current {query_key}?")
    lines.append(f"Which fact IDs (if any) contained stale information before the update?")
    lines.append("")
    lines.append('Respond with ONLY this JSON (no other text):')
    lines.append('{"stale_ids": ["<id>", ...], "final_answer": "<value>"}')

    prompt = "\n".join(lines)
    expected = {
        "stale_ids": stale_ids,
        "final_answer": expected_answer,
    }
    return prompt, expected


def build_source_conflict_prompt(domain: dict, inject_fault: bool = True) -> tuple[str, dict]:
    """Build a source-conflict trial. Randomizes source presentation order."""
    sa = domain["source_a"]
    sb = domain["source_b"]

    # Randomize presentation order to prevent position bias
    sources = [sa, sb]
    random.shuffle(sources)
    first, second = sources

    lines = [f"Context: {domain['context']}. Two sources provide information:"]

    if inject_fault:
        # Show conflicting values
        lines.append(f"\nSource 1 — {first['name']}:")
        lines.append(f"  [{first['fact_id']}] {first['key']}: {first['value']}")
        lines.append(f"\nSource 2 — {second['name']}:")
        lines.append(f"  [{second['fact_id']}] {second['key']}: {second['value']}")

        auth_source = sb if domain["authoritative"] == "B" else sa
        stale_source = sa if domain["authoritative"] == "B" else sb
        # Indirect authority cues — let model infer which source is authoritative
        auth_label = domain.get("auth_label", "Official verified records")
        stale_label = domain.get("stale_label", "Unverified personal notes")
        auth_date = domain.get("auth_date", "March 12, 2026")
        stale_date = domain.get("stale_date", "January 5, 2025")
        lines.append(f"\nAdditional context:")
        lines.append(f"  {auth_source['name']}: Classification — {auth_label}. "
                      f"Last updated {auth_date}.")
        lines.append(f"  {stale_source['name']}: Classification — {stale_label}. "
                      f"Last updated {stale_date}.")
        expected_answer = auth_source["value"]
        stale_ids = [stale_source["fact_id"]]
        support_id = auth_source["fact_id"]
    else:
        # Control: both sources show the SAME value (authoritative value)
        auth_source = sb if domain["authoritative"] == "B" else sa
        agreed_value = auth_source["value"]
        lines.append(f"\nSource 1 — {first['name']}:")
        lines.append(f"  [{first['fact_id']}] {first['key']}: {agreed_value}")
        lines.append(f"\nSource 2 — {second['name']}:")
        lines.append(f"  [{second['fact_id']}] {second['key']}: {agreed_value}")
        lines.append(f"\nNote: Both sources provide consistent information.")
        expected_answer = agreed_value
        stale_ids = []
        support_id = None  # either source is acceptable in control

    # Inject filler turns between authority cues / consistency note and query
    _filler_rng = random.Random(domain["context"] + "_source")
    fillers = _sample_fillers(_filler_rng, n=3)
    for filler in fillers:
        lines.append(f"\n{filler}")
    lines.append("")

    lines.append(f"Based on all the information above, what is the current {domain['query_key']}?")
    lines.append(f"Which fact ID(s) contained outdated information?")
    lines.append(f"Which fact ID supports the correct answer?")
    lines.append("")
    lines.append('Respond with ONLY this JSON (no other text):')
    lines.append('{"stale_ids": ["<id>", ...], "support_id": "<id>", "final_answer": "<value>"}')

    prompt = "\n".join(lines)
    expected = {
        "stale_ids": stale_ids,
        "support_id": support_id,
        "final_answer": expected_answer,
    }
    return prompt, expected


# ── Scoring ─────────────────────────────────────────────────────────────────

def parse_json_response(response: str) -> dict | None:
    """Extract JSON from model response, handling markdown fences."""
    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code fence
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first { ... }
    match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _normalize_answer(s: str) -> str:
    """Normalize an answer string for strict comparison."""
    s = s.strip().lower()
    # Remove quotes
    s = re.sub(r"[\"'`]", "", s)
    # Strip trailing punctuation (period, comma, semicolon)
    s = re.sub(r"[.,;]+$", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def score_trial(parsed: dict | None, expected: dict, trial_type: str) -> dict:
    """Score a single trial. Returns dict with detection, source, recovery scores."""
    if parsed is None:
        return {
            "detection": 0.0,
            "source": 0.0 if trial_type == "source_conflict" else None,
            "recovery": 0.0,
            "composite": 0.0,
            "parse_error": True,
        }

    # Detection: exact match on stale_ids (order-independent)
    pred_stale = set(str(x).upper() for x in parsed.get("stale_ids", []))
    true_stale = set(str(x).upper() for x in expected.get("stale_ids", []))
    detection = 1.0 if pred_stale == true_stale else 0.0

    # Source resolution (ONLY for faulted source-conflict trials with a defined support_id)
    if trial_type == "source_conflict" and expected.get("support_id") is not None:
        pred_support = str(parsed.get("support_id", "")).upper()
        true_support = str(expected["support_id"]).upper()
        source = 1.0 if pred_support == true_support else 0.0
    else:
        source = None  # not applicable (non-source trials or control)

    # Recovery: strict normalized exact match
    pred_answer = _normalize_answer(str(parsed.get("final_answer", "")))
    true_answer = _normalize_answer(str(expected["final_answer"]))
    recovery = 1.0 if pred_answer == true_answer else 0.0

    # Family-specific composite (no free source bonus for non-source trials)
    if trial_type == "source_conflict" and source is not None:
        # 0.35*Detection + 0.15*Source + 0.50*Recovery
        composite = 0.35 * detection + 0.15 * source + 0.50 * recovery
    else:
        # 0.40*Detection + 0.60*Recovery (renormalized without source)
        composite = 0.40 * detection + 0.60 * recovery

    return {
        "detection": detection,
        "source": source,
        "recovery": recovery,
        "composite": composite,
        "parse_error": False,
    }


# ── Trial runner ────────────────────────────────────────────────────────────

def run_emc_lite(
    model_name: str,
    n_per_family: int = 12,
    include_control: bool = True,
    seed: int = RANDOM_SEED,
) -> dict:
    """Run EMC-lite for one model.

    Args:
        model_name: Key into MODELS registry.
        n_per_family: Number of faulted trials per family (max 12).
        include_control: If True, run matched no-fault controls.
        seed: Random seed for stimulus selection.
    """
    random.seed(seed)
    n_per_family = min(n_per_family, 12)

    # Select stimuli
    stale_domains = random.sample(STALE_UPDATE_DOMAINS, n_per_family)
    cross_domains = random.sample(CROSS_EPISODE_DOMAINS, n_per_family)
    source_domains = random.sample(SOURCE_CONFLICT_DOMAINS, n_per_family)

    all_trials = []
    conditions = [True]  # faulted
    if include_control:
        conditions.append(False)  # no-fault control

    builders = [
        ("stale_update", stale_domains, build_stale_update_prompt),
        ("cross_episode", cross_domains, build_cross_episode_prompt),
        ("source_conflict", source_domains, build_source_conflict_prompt),
    ]

    for family_name, domains, builder_fn in builders:
        for inject_fault in conditions:
            condition_label = "faulted" if inject_fault else "control"
            for idx, domain in enumerate(domains):
                prompt, expected = builder_fn(domain, inject_fault=inject_fault)

                try:
                    response = call_model(model_name, prompt, system=SYSTEM_PROMPT)
                    parsed = parse_json_response(response)
                    # Retry once on parse failure with repair prompt
                    if parsed is None:
                        repair_prompt = (
                            "Your previous response was not valid JSON. "
                            "Please respond with ONLY the JSON object, nothing else:\n"
                            + response[:500]
                        )
                        response2 = call_model(model_name, repair_prompt, system=SYSTEM_PROMPT)
                        parsed = parse_json_response(response2)
                        response = response + "\n[RETRY]\n" + response2
                    scores = score_trial(parsed, expected, family_name)
                except Exception as e:
                    response = f"ERROR: {e}"
                    parsed = None
                    scores = score_trial(None, expected, family_name)

                trial = {
                    "family": family_name,
                    "condition": condition_label,
                    "trial_idx": idx,
                    "context": domain["context"],
                    "inject_fault": inject_fault,
                    "raw_response": response,  # full response for diagnosis
                    "parsed": parsed,
                    "expected": expected,
                    **scores,
                }
                all_trials.append(trial)
                time.sleep(0.3)

    # Aggregate scores
    faulted = [t for t in all_trials if t["condition"] == "faulted"]
    control = [t for t in all_trials if t["condition"] == "control"]

    def _agg(trials):
        if not trials:
            return {}
        source_vals = [t["source"] for t in trials if t["source"] is not None]
        return {
            "detection": float(np.mean([t["detection"] for t in trials])),
            "source": float(np.mean(source_vals)) if source_vals else None,
            "recovery": float(np.mean([t["recovery"] for t in trials])),
            "composite": float(np.mean([t["composite"] for t in trials])),
            "parse_error_rate": float(np.mean([t["parse_error"] for t in trials])),
            "n": len(trials),
        }

    def _agg_by_family(trials):
        families = {}
        for fam in ["stale_update", "cross_episode", "source_conflict"]:
            fam_trials = [t for t in trials if t["family"] == fam]
            families[fam] = _agg(fam_trials)
        return families

    result = {
        "model": model_name,
        "seed": seed,
        "n_per_family": n_per_family,
        "include_control": include_control,
        "timestamp": datetime.utcnow().isoformat(),
        "faulted": _agg(faulted),
        "control": _agg(control),
        "faulted_by_family": _agg_by_family(faulted),
        "control_by_family": _agg_by_family(control),
        "trials": all_trials,
    }

    # Completion proxy: control recovery rate (should be ~0.90+)
    if control:
        result["control_completion"] = result["control"]["recovery"]

    return result


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EMC-lite: Fault Detection & Recovery Probes")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--all-ollama", action="store_true")
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot mode: 3 models (highest/median/lowest WMF-AM)")
    parser.add_argument("--n-per-family", type=int, default=12)
    parser.add_argument("--no-control", action="store_true",
                        help="Skip no-fault control condition")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Select models
    if args.pilot:
        # Highest, median, lowest WMF-AM from N=7 pilot
        selected = [
            "ollama:deepseek-r1:14b",  # WMF-AM=0.983 (highest)
            "ollama:qwen2.5:7b",       # WMF-AM=0.350 (median)
            "ollama:llama3.1:8b",      # WMF-AM=0.183 (lowest)
        ]
    elif args.all_ollama:
        selected = [m for m in MODELS if MODELS[m]["provider"] == "ollama"
                     and "70b" not in m]  # skip 70b (timeout issues)
    elif args.models:
        selected = args.models
    else:
        selected = [
            "ollama:deepseek-r1:14b",
            "ollama:qwen2.5:7b",
            "ollama:qwen2.5:14b",
            "ollama:qwen2.5:32b",
            "ollama:llama3.1:8b",
            "ollama:gemma2:27b",
            "ollama:mistral:7b",
        ]

    include_control = not args.no_control
    n_trials = args.n_per_family * 3 * (2 if include_control else 1)

    print(f"EMC-lite — {len(selected)} models × {n_trials} trials/model", flush=True)
    print(f"Models: {selected}", flush=True)
    print(f"Seed: {args.seed}  Control: {include_control}", flush=True)
    print(f"Started: {datetime.utcnow().isoformat()}\n", flush=True)

    all_results = []
    for model in selected:
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {model}", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()
        result = run_emc_lite(
            model, n_per_family=args.n_per_family,
            include_control=include_control, seed=args.seed,
        )
        elapsed = time.time() - t0
        all_results.append(result)

        f = result["faulted"]
        c = result.get("control", {})
        f_src_str = f"{f['source']:.3f}" if f['source'] is not None else "N/A"
        print(f"  Faulted:  det={f['detection']:.3f}  src={f_src_str}  "
              f"rec={f['recovery']:.3f}  comp={f['composite']:.3f}  "
              f"parse_err={f['parse_error_rate']:.2f}", flush=True)
        if c:
            c_src_str = f"{c['source']:.3f}" if c['source'] is not None else "N/A"
            print(f"  Control:  det={c['detection']:.3f}  src={c_src_str}  "
                  f"rec={c['recovery']:.3f}  comp={c['composite']:.3f}  "
                  f"parse_err={c['parse_error_rate']:.2f}", flush=True)
        print(f"  ({elapsed:.0f}s)", flush=True)

    # Summary table
    print(f"\n\n{'='*70}", flush=True)
    print("EMC-LITE RESULTS SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Model':<25} {'F-Det':>6} {'F-Src':>6} {'F-Rec':>6} {'F-Comp':>7} "
          f"{'C-Rec':>6} {'ΔRec':>6} {'Parse%':>7}", flush=True)
    print("-" * 75, flush=True)
    for r in all_results:
        f = r["faulted"]
        c = r.get("control", {})
        c_rec = c.get("recovery", float("nan"))
        delta_rec = f["recovery"] - c_rec if c else float("nan")
        f_src = f.get("source")
        f_src_str = f"{f_src:>6.3f}" if f_src is not None else "   N/A"
        model_short = r["model"].replace("ollama:", "")
        print(f"{model_short:<25} {f['detection']:>6.3f} {f_src_str} "
              f"{f['recovery']:>6.3f} {f['composite']:>7.3f} "
              f"{c_rec:>6.3f} {delta_rec:>+6.3f} {f['parse_error_rate']:>7.1%}",
              flush=True)

    # Check pilot success criteria
    if len(all_results) >= 3:
        composites = [r["faulted"]["composite"] for r in all_results]
        spread = max(composites) - min(composites)
        ctrl_recs = [r["control"]["recovery"] for r in all_results if r.get("control")]
        mean_ctrl = np.mean(ctrl_recs) if ctrl_recs else float("nan")

        print(f"\nPilot diagnostics:", flush=True)
        print(f"  EMC-lite spread (max-min composite): {spread:.3f} (target ≥ 0.20)", flush=True)
        print(f"  Mean control recovery: {mean_ctrl:.3f} (target ≥ 0.85)", flush=True)

        faulted_recs = [r["faulted"]["recovery"] for r in all_results]
        ctrl_recs_matched = [r["control"]["recovery"] for r in all_results if r.get("control")]
        if ctrl_recs_matched:
            completion_drop = np.mean(ctrl_recs_matched) - np.mean(faulted_recs)
            print(f"  Control-Faulted recovery drop: {completion_drop:.3f} (expect small but >0)", flush=True)

    # Save
    output = {
        "experiment": "emc_lite",
        "timestamp": datetime.utcnow().isoformat(),
        "models": selected,
        "n_models": len(all_results),
        "seed": args.seed,
        "n_per_family": args.n_per_family,
        "include_control": include_control,
        "per_model": all_results,
    }
    out_path = args.output or str(RESULTS_DIR / f"emc_lite_{args.seed}.json")
    Path(out_path).write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}", flush=True)
    print(f"Finished: {datetime.utcnow().isoformat()}", flush=True)


if __name__ == "__main__":
    main()
