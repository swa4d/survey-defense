#!/usr/bin/env python3
"""
Bot Detection CLI
-----------------
Usage:
    python run_bot_detection.py responses.csv
    python run_bot_detection.py responses.csv --threshold 0.6
    python run_bot_detection.py responses.csv --output results.csv --threshold 0.5
    python run_bot_detection.py responses.csv --train train.csv --predict new_responses.csv

Options:
    --output FILE       Save results to CSV (default: <input>_scored.csv)
    --threshold FLOAT   Bot probability threshold (default: 0.5)
    --train FILE        Separate training CSV (fit on this, score the main input)
    --json              Also save a .json file of results
    --summary           Print summary stats only (no per-row output)
    --top N             Show top N most likely bots in terminal (default: 10)

The script assumes bot_detector.py is in the same directory.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Import the detection system ───────────────────────────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
try:
    from bot_detector import BotDetectionPipeline, ValidationFramework
except ImportError:
    print("ERROR: bot_detector.py not found. Place it in the same directory as this script.")
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV with basic validation and diagnostics."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    if p.suffix.lower() != ".csv":
        print(f"WARNING: Expected .csv file, got {p.suffix}")

    print(f"Loading {p.name}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # Warn about missing key columns
    key_cols = [
        "Duration (in seconds)", "auto_webdriver", "auto_score",
        "Q_RecaptchaScore", "captcha_pass",
    ]
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing expected columns: {missing}")
        print(f"  The system will still run but some signals will be unavailable.")

    if "ResponseId" not in df.columns:
        print("  NOTE: No 'ResponseId' column found — using row index as ID.")
        df["ResponseId"] = df.index.astype(str)

    return df


def results_to_dataframe(results) -> pd.DataFrame:
    """Flatten results list into a tidy DataFrame."""
    rows = []
    for r in results:
        row = {
            "ResponseId": r.ResponseId,
            "bot_probability": r.bot_probability,
            "final_label": r.final_label,
            "final_label_text": "BOT" if r.final_label == 1 else "HUMAN",
            "confidence": r.confidence,
        }
        row.update({f"subscore_{k}": v for k, v in r.subscores.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def print_banner(text: str) -> None:
    w = 62
    print("\n" + "═" * w)
    print(f"  {text}")
    print("═" * w)


def print_summary(results_df: pd.DataFrame, elapsed: float) -> None:
    """Print a human-readable summary to the terminal."""
    n = len(results_df)
    n_bot = (results_df["final_label"] == 1).sum()
    n_human = (results_df["final_label"] == -1).sum()
    high_conf = (results_df["confidence"] > 0.8).sum()

    print_banner("DETECTION SUMMARY")
    print(f"  {'Responses scored:':<30} {n:,}")
    print(f"  {'Flagged as BOT:':<30} {n_bot:,}  ({n_bot/n:.1%})")
    print(f"  {'Flagged as HUMAN:':<30} {n_human:,}  ({n_human/n:.1%})")
    print(f"  {'High-confidence (>0.8):':<30} {high_conf:,}  ({high_conf/n:.1%})")
    print(f"  {'Mean bot probability:':<30} {results_df['bot_probability'].mean():.3f}")
    print(f"  {'Processing time:':<30} {elapsed:.1f}s  ({n/elapsed:.0f} responses/sec)")

    print_banner("SUBSCORE AVERAGES  (bots vs humans)")
    subscore_cols = [c for c in results_df.columns if c.startswith("subscore_")]
    bots = results_df[results_df["final_label"] == 1]
    hums = results_df[results_df["final_label"] == -1]
    for col in subscore_cols:
        name = col.replace("subscore_", "").replace("_", " ")
        bot_mean = bots[col].mean() if len(bots) else float("nan")
        hum_mean = hums[col].mean() if len(hums) else float("nan")
        print(f"  {name:<22} BOT: {bot_mean:.3f}   HUMAN: {hum_mean:.3f}")

    print_banner("CONFIDENCE DISTRIBUTION")
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.01]
    labels = ["<0.5 (uncertain)", "0.5–0.7 (low)", "0.7–0.8 (moderate)",
              "0.8–0.9 (high)", "0.9–1.0 (very high)"]
    cuts = pd.cut(results_df["confidence"], bins=bins, labels=labels, right=False)
    for label, count in cuts.value_counts().sort_index().items():
        bar = "█" * int(count / n * 40)
        print(f"  {label:<25} {count:>5}  {bar}")


def print_top_bots(results_df: pd.DataFrame, n: int) -> None:
    """Print the top N highest-probability bot responses."""
    print_banner(f"TOP {n} MOST LIKELY BOTS")
    top = results_df.nlargest(n, "bot_probability")
    subscore_cols = [c for c in results_df.columns if c.startswith("subscore_")]
    for _, row in top.iterrows():
        conf_str = f"{row['confidence']:.2f}"
        print(f"\n  ResponseId: {row['ResponseId']}")
        print(f"    Bot probability : {row['bot_probability']:.4f}  |  Confidence: {conf_str}")
        for col in subscore_cols:
            name = col.replace("subscore_", "")
            val = row[col]
            bar = "█" * int(val * 20)
            print(f"    {name:<22}: {val:.3f}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score survey responses for bot activity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="CSV file of survey responses to score")
    parser.add_argument("--output", help="Output CSV path (default: <input>_scored.csv)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Bot probability threshold (default: 0.5)")
    parser.add_argument("--train", help="Separate training CSV (fit on this, score input)")
    parser.add_argument("--json", action="store_true", help="Also save JSON output")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary stats only, skip per-row terminal output")
    parser.add_argument("--top", type=int, default=10,
                        help="Show top N bot results in terminal (default: 10, 0 to disable)")

    args = parser.parse_args()

    # ── Validate threshold ────────────────────────────────────────────────────
    if not 0.0 < args.threshold < 1.0:
        print(f"ERROR: --threshold must be between 0 and 1, got {args.threshold}")
        sys.exit(1)

    # ── Output path ───────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_scored.csv"

    # ── Load data ─────────────────────────────────────────────────────────────
    print_banner("BOT DETECTION SYSTEM")
    df_score = load_csv(args.input)

    if args.train:
        df_train = load_csv(args.train)
        print(f"\nMode: Train on '{args.train}', score '{args.input}'")
    else:
        df_train = df_score
        print(f"\nMode: Fit-and-score on '{args.input}'")

    print(f"Threshold: {args.threshold}  (responses with P(bot) ≥ {args.threshold} → BOT)")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    print_banner("RUNNING PIPELINE")
    t0 = time.time()

    pipeline = BotDetectionPipeline(bot_threshold=args.threshold)

    if args.train:
        print("\nFitting pipeline on training data...")
        pipeline.fit(df_train)
        print("\nScoring target responses...")
        results = pipeline.predict(df_score)
    else:
        print("\nFitting and scoring...")
        results = pipeline.fit_predict(df_score)

    elapsed = time.time() - t0

    # ── Format results ────────────────────────────────────────────────────────
    results_df = results_to_dataframe(results)

    # Optionally merge back with original data
    scored_df = df_score.copy()
    scored_df = scored_df.merge(
        results_df.drop(columns=["final_label_text"], errors="ignore"),
        on="ResponseId",
        how="left",
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    print_banner("SAVING OUTPUT")
    results_df.to_csv(output_path, index=False)
    print(f"  Scores saved:  {output_path}")

    # Merged version (original data + scores)
    merged_path = output_path.parent / f"{output_path.stem}_merged.csv"
    scored_df.to_csv(merged_path, index=False)
    print(f"  Merged saved:  {merged_path}")

    if args.json:
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"  JSON saved:    {json_path}")

    # ── Terminal output ───────────────────────────────────────────────────────
    print_summary(results_df, elapsed)

    if not args.summary and args.top > 0:
        print_top_bots(results_df, min(args.top, len(results_df)))

    # ── Validation agreement metrics ──────────────────────────────────────────
    validator = ValidationFramework()
    agreement = validator.compute_agreement_metrics(results)
    print_banner("SIGNAL AGREEMENT  (quality indicator)")
    print(f"  {'Strong multi-signal BOT:':<32} {agreement['strong_bot_pct']:.1%}")
    print(f"  {'Strong multi-signal HUMAN:':<32} {agreement['strong_human_pct']:.1%}")
    print(f"  {'Contradictory signals:':<32} {agreement['contradictory_pct']:.1%}")
    print(f"  {'Mean confidence:':<32} {agreement['mean_confidence']:.3f}")

    contradictory_pct = agreement["contradictory_pct"]
    if contradictory_pct > 0.20:
        print(f"\n  ⚠  {contradictory_pct:.0%} of responses have contradictory signals.")
        print(f"     Consider reviewing flagged responses manually.")

    print(f"\n  Done. Output: {output_path}\n")


if __name__ == "__main__":
    main()