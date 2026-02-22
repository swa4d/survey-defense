"""
Qualtrics Survey Bot Detection - Ensemble Classifier System
============================================================
Vote convention:
  +1 = bot,  -1 = human,  confidence in [0, 1],  -1 = abstain (no data)

Final score per respondent:
  raw_score  = sig_c  w_c x vote_c x conf_c          (signed: positive = bot)
  norm_score = raw_score / Σ_c  w_c x |conf_c|      (bounded [-1, +1])
  final_vote = +1 if norm_score > 0 else -1

Using signed votes means confidence CORRECTLY pulls scores toward both
poles: a high-confidence human vote (-1 x 0.9) actively cancels a
weak bot vote (+1 x 0.3) rather than simply adding nothing.
"""

import pandas as pd
import numpy as np
import warnings
import json
import re
import argparse
import sys
from pathlib import Path
from typing import Optional
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEIGHTS = {
    "no_keystrokes":         0.12,
    "no_click_counts":       0.08,
    "recaptcha":             0.16,
    "automation_flags":      0.12,
    "uniform_timing":        0.08,
    "big_paste":             0.08,
    "low_keys_high_text":    0.08,
    "random_forest_outlier": 0.08,
    "generic_tags":          0.04,
    "text_burstiness":       0.04,
    "text_perplexity_proxy": 0.04,
    "text_repetition":       0.04,
    "text_ai_vocabulary":    0.04,
    "text_coherence":        0.04,
    "text_structure":        0.04,
}

# QIDs that collect open-ended text responses (tracked by HumanTipper)
TEXT_QIDS = ["QID584", "QID629", "QID631", "QID690", "QID694"]

# Map QID -> CSV column name holding the written answer.
# Populate with your actual Qualtrics column headers. Example:
#   TEXT_RESPONSE_COLS = {"QID584": "Q1_TEXT", "QID629": "Q2_TEXT"}
TEXT_RESPONSE_COLS: dict = {}

# Click-tracked question stems
CLICK_STEMS = ["gender", "hispanic", "race"]

# Impossibly fast completion threshold (seconds)
IMPOSSIBLE_TIME_THRESHOLD = 30

# Common AI vocabulary/phrase fingerprints — regex patterns
AI_VOCABULARY_SIGNALS = [
    r"\bcertainly\b", r"\bsure(ly)?\b", r"\babsolutely\b", r"\bof course\b",
    r"\bit'?s important to\b", r"\bit is worth (noting|mentioning)\b",
    r"\bin conclusion\b", r"\bfurthermore\b", r"\bmoreover\b", r"\bin summary\b",
    r"\bI (would|want to) (note|highlight|emphasize|mention)\b",
    r"\bas an AI\b", r"\bas a language model\b",
    r"\bI (do not|don'?t) have (personal|the ability|access)\b",
    r"\bI (hope|trust) (this|that) helps?\b",
    r"\bplease (let me know|feel free)\b",
    r"\bI'?m happy to\b", r"\bI'?d be (happy|glad|delighted) to\b",
    r"\bthank you for (asking|your question|reaching out)\b",
    r"\bgreat question\b",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _get_text(df: pd.DataFrame, rid, qid: str) -> Optional[str]:
    col = TEXT_RESPONSE_COLS.get(qid)
    if col and col in df.columns:
        v = df.at[rid, col]
        return str(v).strip() if pd.notna(v) and str(v).strip() else None
    return None


def _all_texts(df: pd.DataFrame, rid) -> list:
    texts = []
    for qid in TEXT_QIDS:
        t = _get_text(df, rid, qid)
        if t:
            texts.append(t)
    return texts


def _vote(is_bot: bool, conf: float) -> tuple:
    """Return (vote, conf) using the +1/-1 convention."""
    return (1, conf) if is_bot else (-1, conf)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """
    Qualtrics CSV: row 0 = headers, rows 1-2 = label/ImportID metadata.
    Skip rows 1 and 2 so actual data starts at row 3.
    """
    df = pd.read_csv(path, skiprows=[1, 2], low_memory=False)
    df.reset_index(drop=True, inplace=True)
    if "ResponseId" in df.columns:
        df.set_index("ResponseId", inplace=True)
    else:
        df.index = [f"R{i+1}" for i in range(len(df))]
    return df


def identify_assured_bots(df: pd.DataFrame) -> pd.Index:
    dur = _safe_numeric(df, "Q_TotalDuration")
    return df.index[dur < IMPOSSIBLE_TIME_THRESHOLD]

# ===========================================================================
# Behavioral classifiers (keystroke / automation signals)
# ===========================================================================

def clf_no_keystrokes(df: pd.DataFrame):
    """Zero keystrokes across all text fields -> strong bot signal."""
    results = []
    cols = [f"ht_keydown_{qid}" for qid in TEXT_QIDS if f"ht_keydown_{qid}" in df.columns]
    if not cols:
        return results
    for rid in df.index:
        vals = [pd.to_numeric(df.at[rid, c], errors="coerce") for c in cols]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            results.append((rid, 0, -1)); continue
        total = sum(vals)
        if total == 0:
            results.append((rid, *_vote(True, 0.95)))
        elif total < 5:
            results.append((rid, *_vote(True, 0.60)))
        else:
            results.append((rid, *_vote(False, 0.80)))
    return results


def clf_no_click_counts(df: pd.DataFrame):
    """Zero clicks on all click-tracked questions suggests automation."""
    results = []
    cols = [f"{s}_Click Count" for s in CLICK_STEMS if f"{s}_Click Count" in df.columns]
    if not cols:
        return results
    for rid in df.index:
        vals = [pd.to_numeric(df.at[rid, c], errors="coerce") for c in cols]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            results.append((rid, 0, -1)); continue
        if sum(vals) == 0:
            results.append((rid, *_vote(True, 0.85)))
        else:
            results.append((rid, *_vote(False, 0.75)))
    return results


def clf_recaptcha(df: pd.DataFrame):
    """Q_RecaptchaScore: 0.0=bot, 1.0=human. Confidence scales with extremity."""
    results = []
    scores = _safe_numeric(df, "Q_RecaptchaScore")
    for rid in df.index:
        s = scores[rid]
        if np.isnan(s):
            results.append((rid, 0, -1)); continue
        if s <= 0.3:
            results.append((rid, *_vote(True,  round(1.0 - s, 4))))
        elif s >= 0.7:
            results.append((rid, *_vote(False, round(s, 4))))
        else:
            is_bot = s < 0.5
            conf   = round(abs(s - 0.5) * 2 * 0.5, 4)
            results.append((rid, *_vote(is_bot, conf)))
    return results


def clf_automation_flags(df: pd.DataFrame):
    """Composite of auto_* browser fingerprint signals."""
    results = []
    binary  = ["auto_webdriver", "auto_phantom", "auto_nightmare", "auto_domAutomation"]
    avail   = [c for c in binary if c in df.columns]
    ascore  = _safe_numeric(df, "auto_score")
    plugins = _safe_numeric(df, "auto_pluginsLen")
    langs   = _safe_numeric(df, "auto_langsLen")
    captcha = _safe_numeric(df, "captcha_pass")
    for rid in df.index:
        score, n = 0.0, 0
        for col in avail:
            v = pd.to_numeric(df.at[rid, col], errors="coerce")
            if not np.isnan(v): score += float(v); n += 1
        if not np.isnan(ascore[rid]):
            score += ascore[rid] / 100.0; n += 1
        if not np.isnan(plugins[rid]):
            score += 0.75 if plugins[rid] == 0 else 0.0; n += 1
        if not np.isnan(langs[rid]):
            score += 0.60 if langs[rid] == 0 else 0.0; n += 1
        if not np.isnan(captcha[rid]):
            score += 0.90 if captcha[rid] == 0 else 0.0; n += 1
        if n == 0:
            results.append((rid, 0, -1)); continue
        norm   = score / n
        is_bot = norm > 0.4
        conf   = round(min(norm * 1.5 if is_bot else (1 - norm) * 1.5, 1.0), 4)
        results.append((rid, *_vote(is_bot, conf)))
    return results


def clf_uniform_timing(df: pd.DataFrame):
    """ht_flag_uniformTiming: programmatic regularity in inter-key timing."""
    results = []
    cols = [f"ht_flag_uniformTiming_{qid}" for qid in TEXT_QIDS
            if f"ht_flag_uniformTiming_{qid}" in df.columns]
    if not cols:
        return results
    for rid in df.index:
        flagged, total = 0, 0
        for c in cols:
            v = pd.to_numeric(df.at[rid, c], errors="coerce")
            if not np.isnan(v):
                total += 1
                if v == 1: flagged += 1
        if total == 0:
            results.append((rid, 0, -1)); continue
        ratio = flagged / total
        if ratio >= 0.6:
            results.append((rid, *_vote(True,  round(min(0.80 + ratio * 0.15, 0.95), 4))))
        elif ratio > 0:
            results.append((rid, *_vote(True,  round(0.50 + ratio * 0.30, 4))))
        else:
            results.append((rid, *_vote(False, 0.70)))
    return results


def clf_big_paste(df: pd.DataFrame):
    """ht_flag_bigPaste + raw paste/totalChars ratio."""
    results = []
    for rid in df.index:
        flag_c, flag_t, sigs = 0, 0, []
        for qid in TEXT_QIDS:
            fc = f"ht_flag_bigPaste_{qid}"
            pc = f"ht_pastedChars_{qid}"
            tc = f"ht_totalChars_{qid}"
            if fc in df.columns:
                v = pd.to_numeric(df.at[rid, fc], errors="coerce")
                if not np.isnan(v): flag_t += 1; flag_c += int(v == 1)  # noqa: E702
            if pc in df.columns and tc in df.columns:
                p = pd.to_numeric(df.at[rid, pc], errors="coerce")
                t = pd.to_numeric(df.at[rid, tc], errors="coerce")
                if not np.isnan(p) and not np.isnan(t) and t > 0:
                    sigs.append(p / t)
        parts = []
        if flag_t: parts.append(flag_c / flag_t)
        if sigs:   parts.append(np.mean(sigs))
        if not parts:
            results.append((rid, 0, -1)); continue
        composite = np.mean(parts)
        if composite >= 0.6:
            results.append((rid, *_vote(True,  round(min(0.5 + composite * 0.5, 0.95), 4))))
        elif composite > 0.3:
            results.append((rid, *_vote(True,  round(composite, 4))))
        else:
            results.append((rid, *_vote(False, round(1.0 - composite, 4))))
    return results


def clf_low_keys_high_text(df: pd.DataFrame):
    """ht_flag_lowKeysHighText + direct keystroke-to-char ratio."""
    results = []
    for rid in df.index:
        flag_c, flag_t, sigs = 0, 0, []
        for qid in TEXT_QIDS:
            fc = f"ht_flag_lowKeysHighText_{qid}"
            kc = f"ht_keydown_{qid}"
            tc = f"ht_totalChars_{qid}"
            if fc in df.columns:
                v = pd.to_numeric(df.at[rid, fc], errors="coerce")
                if not np.isnan(v): flag_t += 1; flag_c += int(v == 1)  # noqa: E702
            if kc in df.columns and tc in df.columns:
                k = pd.to_numeric(df.at[rid, kc], errors="coerce")
                t = pd.to_numeric(df.at[rid, tc], errors="coerce")
                if not np.isnan(k) and not np.isnan(t) and t > 0:
                    sigs.append(1.0 if k < t * 0.5 else 0.0)
        parts = []
        if flag_t:     parts.append(flag_c / flag_t)
        if sigs:       parts.append(np.mean(sigs))
        if not parts:
            results.append((rid, 0, -1)); continue
        score  = np.mean(parts)
        is_bot = score >= 0.5
        conf   = round(min(0.4 + score * 0.6 if is_bot else 1.0 - score, 0.95), 4)
        results.append((rid, *_vote(is_bot, conf)))
    return results


def clf_random_forest_outlier(df: pd.DataFrame, assured_bots: pd.Index):
    """IsolationForest (40%) + semi-supervised RandomForest (60%) on all numeric HT/auto features."""
    results = []
    feat_cols = [c for c in df.columns if c.startswith(("ht_", "auto_", "Q_Recaptcha"))]
    if not feat_cols:
        return [(rid, 0, -1) for rid in df.index]

    X_raw = df[feat_cols].copy().apply(pd.to_numeric, errors="coerce")
    pipe  = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    X     = pipe.fit_transform(X_raw)
    n     = len(df)

    labels = np.zeros(n, dtype=int)
    ab_pos = [i for i, rid in enumerate(df.index) if rid in assured_bots]
    for p in ab_pos: labels[p] = 1

    iso    = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    iso_sc = iso.fit(X).score_samples(X)
    lo, hi = iso_sc.min(), iso_sc.max()
    iso_p  = 1.0 - (iso_sc - lo) / (hi - lo + 1e-9)

    if len(ab_pos) >= 2:
        rf    = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
        rf.fit(X, labels)
        rf_p  = rf.predict_proba(X)[:, 1]
    else:
        rf_p  = np.zeros(n)

    iso_w = 0.4
    rf_w  = 0.6 if len(ab_pos) >= 2 else 0.0
    denom = iso_w + rf_w if rf_w > 0 else iso_w

    for i, rid in enumerate(df.index):
        prob   = (iso_w * iso_p[i] + rf_w * rf_p[i]) / denom
        is_bot = prob > 0.5
        conf   = float(min(abs(prob - 0.5) * 2, 0.95))
        results.append((rid, *_vote(is_bot, round(conf, 4))))
    return results


def clf_generic_tags(df: pd.DataFrame):
    """Q_BallotBoxStuffing, Q_TerminateFlag, Page Submit timing."""
    results = []
    bbs    = _safe_numeric(df, "Q_BallotBoxStuffing")
    term   = _safe_numeric(df, "Q_TerminateFlag")
    s_cols = [f"{s}_Page Submit" for s in CLICK_STEMS if f"{s}_Page Submit" in df.columns]
    for rid in df.index:
        score, n = 0.0, 0
        v = bbs[rid]
        if not np.isnan(v): score += float(v); n += 1
        vt = term[rid]
        if not np.isnan(vt) and vt == 1: score += 0.3; n += 1
        for sc in s_cols:
            t = _safe_numeric(df, sc)[rid]
            if not np.isnan(t): score += 0.8 if t < 2 else 0.0; n += 1
        if n == 0:
            results.append((rid, 0, -1)); continue
        norm   = score / n
        is_bot = norm > 0.4
        conf   = round(min(norm * 1.2 if is_bot else (1 - norm) * 1.2, 0.95), 4)
        results.append((rid, *_vote(is_bot, conf)))
    return results


# ===========================================================================
# TEXT ANALYSIS CLASSIFIERS
#
# All six text classifiers operate on the actual written survey responses.
# Configure TEXT_RESPONSE_COLS at the top to map QID -> CSV column name.
# Each classifier gracefully abstains (conf=-1) when no text is available.
# ===========================================================================

def clf_text_burstiness(df: pd.DataFrame):
    """
    Human writing is bursty: sentence lengths vary wildly.
    AI writing is uniform: medium-length sentences throughout.

    Metric: coefficient of variation (CV = std/mean) of per-sentence word counts.
    Low CV (< 0.20) -> high confidence bot.
    High CV (> 0.40) -> human signal.
    """
    results = []
    for rid in df.index:
        texts = _all_texts(df, rid)
        if not texts:
            results.append((rid, 0, -1)); continue

        cvs = []
        for text in texts:
            sents  = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
            if len(sents) < 3: continue
            lens   = np.array([len(s.split()) for s in sents], dtype=float)
            mean_l = lens.mean()
            if mean_l > 0:
                cvs.append(lens.std() / mean_l)

        if not cvs:
            results.append((rid, 0, -1)); continue

        cv = np.mean(cvs)
        if cv < 0.20:
            results.append((rid, *_vote(True,  round(min(0.95, (0.20 - cv) * 5), 4))))
        elif cv < 0.40:
            results.append((rid, *_vote(True,  round((0.40 - cv) * 2.5, 4))))
        else:
            results.append((rid, *_vote(False, round(min((cv - 0.40) * 2, 0.85), 4))))
    return results


def clf_text_perplexity_proxy(df: pd.DataFrame):
    """
    Approximates perplexity using within-dataset bigram commonality.

    AI responses from the same underlying prompt tend to share vocabulary
    and phrasing. A response that uses a disproportionately high fraction
    of the dataset's most common bigrams is suspicious.

    Steps:
    1. Build a bigram frequency table across all responses.
    2. Define "common bigrams" as the top-5% most frequent.
    3. Per respondent, compute the fraction of their bigrams that are common.
    4. Z-score that fraction within the sample. High z -> bot signal.

    This is a population-level signal — most effective with >= 30 respondents.
    """
    results = []
    all_bg: Counter = Counter()
    rid_bg: dict    = {}

    for rid in df.index:
        texts = _all_texts(df, rid)
        combined = " ".join(texts).lower() if texts else ""
        words    = re.findall(r"[a-z']+", combined)
        bgs      = list(zip(words, words[1:]))
        rid_bg[rid] = bgs
        all_bg.update(bgs)

    if not all_bg:
        return [(rid, 0, -1) for rid in df.index]

    cutoff = np.percentile(list(all_bg.values()), 95)
    common = {bg for bg, cnt in all_bg.items() if cnt >= max(cutoff, 2)}

    fracs = []
    for rid in df.index:
        bgs = rid_bg.get(rid, [])
        fracs.append(sum(1 for bg in bgs if bg in common) / len(bgs) if bgs else np.nan)

    valid  = [f for f in fracs if not np.isnan(f)]
    if not valid:
        return [(rid, 0, -1) for rid in df.index]

    mu, sigma = np.mean(valid), np.std(valid) + 1e-9

    for rid, frac in zip(df.index, fracs):
        if np.isnan(frac):
            results.append((rid, 0, -1)); continue
        z = (frac - mu) / sigma
        if z > 1.5:
            results.append((rid, *_vote(True,  round(min(z / 3.0, 0.90), 4))))
        elif z > 0.5:
            results.append((rid, *_vote(True,  round(z * 0.2, 4))))
        elif z < -1.0:
            results.append((rid, *_vote(False, round(min(abs(z) / 3.0, 0.75), 4))))
        else:
            results.append((rid, *_vote(False, 0.30)))
    return results


def clf_text_repetition(df: pd.DataFrame):
    """
    Two repetition signals:
    A) Trigram uniqueness within a single answer (AI often rephrases similarly).
       Low unique-trigram ratio -> high repetition -> bot.
    B) Cross-question Jaccard word-overlap (same vocabulary across different
       questions suggests template reuse or copy-paste).
    """
    results = []
    for rid in df.index:
        texts = _all_texts(df, rid)
        if not texts:
            results.append((rid, 0, -1)); continue

        sigs = []
        # Signal A: internal trigram uniqueness
        for text in texts:
            words    = re.findall(r"[a-z']+", text.lower())
            trigrams = list(zip(words, words[1:], words[2:]))
            if len(trigrams) >= 5:
                sigs.append(1.0 - len(set(trigrams)) / len(trigrams))

        # Signal B: cross-question Jaccard similarity
        if len(texts) >= 2:
            wsets = [set(re.findall(r"[a-z']+", t.lower())) for t in texts]
            jscores = []
            for i in range(len(wsets)):
                for j in range(i + 1, len(wsets)):
                    u = wsets[i] | wsets[j]
                    if u: jscores.append(len(wsets[i] & wsets[j]) / len(u))
            if jscores: sigs.append(np.mean(jscores))

        if not sigs:
            results.append((rid, 0, -1)); continue

        score  = np.mean(sigs)
        if score > 0.6:
            results.append((rid, *_vote(True,  round(min(score * 1.2, 0.90), 4))))
        elif score > 0.35:
            results.append((rid, *_vote(True,  round(score * 0.7, 4))))
        else:
            results.append((rid, *_vote(False, round(0.5 + (0.35 - score), 4))))
    return results


def clf_text_ai_vocabulary(df: pd.DataFrame):
    """
    LLMs have vocabulary tells: hedging phrases, meta-commentary, polite preambles
    that are very rare in authentic survey responses.

    Matches AI_VOCABULARY_SIGNALS regex patterns; confidence scales with match count
    and match density (per ~50-word chunk) to avoid penalizing long genuine responses.
    """
    results = []
    patterns = [re.compile(p, re.IGNORECASE) for p in AI_VOCABULARY_SIGNALS]
    for rid in df.index:
        texts = _all_texts(df, rid)
        if not texts:
            results.append((rid, 0, -1)); continue
        combined  = " ".join(texts)
        n_words   = max(len(combined.split()), 1)
        n_matches = sum(1 for pat in patterns if pat.search(combined))
        density   = n_matches / max(n_words / 50, 1)

        if n_matches >= 4 or density >= 1.5:
            results.append((rid, *_vote(True, round(min(0.40 + n_matches * 0.08, 0.92), 4))))
        elif n_matches >= 2:
            results.append((rid, *_vote(True,  round(0.40 + density * 0.10, 4))))
        elif n_matches == 1:
            results.append((rid, *_vote(True,  0.25)))
        else:
            results.append((rid, *_vote(False, 0.55)))
    return results


def clf_text_coherence(df: pd.DataFrame):
    """
    Measures stylistic proxies for authentic writing without a full NLP model:
    - Average word length: AI tends toward a narrow 4.8-5.8 char band.
    - Non-alphabetic character density: casual human writing has more punctuation,
      numbers, and informal characters than polished AI output.
    - Response length: very short typed answers (< 5 words) are dismissive/bot-like.

    Intentionally capped at 0.60 confidence — this is a weak, corroborating signal.
    """
    results = []
    for rid in df.index:
        texts = _all_texts(df, rid)
        if not texts:
            results.append((rid, 0, -1)); continue
        combined = " ".join(texts)
        words    = re.findall(r"[a-zA-Z']+", combined)
        if len(words) < 5:
            results.append((rid, *_vote(True, 0.35))); continue
        avg_wl    = np.mean([len(w) for w in words])
        non_alpha = len(re.findall(r"[^a-zA-Z\s]", combined)) / max(len(combined), 1)
        sigs      = []
        sigs.append(0.55 if 4.8 <= avg_wl <= 5.8 else 0.20)
        sigs.append(0.15 if non_alpha > 0.10 else 0.45)
        score  = np.mean(sigs)
        is_bot = score > 0.4
        conf   = max(min(round(abs(score - 0.4) * 3, 4), 0.60), 0.10)
        results.append((rid, *_vote(is_bot, conf)))
    return results


def clf_text_structure(df: pd.DataFrame):
    """
    AI responses frequently follow a predictable template:
    opening restatement -> structured body points -> closing summary.

    Detects:
    - All responses end with a period (rigid formatting)
    - Sentence count uniformity across responses (low CV)
    - All responses start with a capital letter (suggests templating)

    Requires >= 2 text responses to compare; abstains otherwise.
    """
    results = []
    for rid in df.index:
        texts = _all_texts(df, rid)
        if len(texts) < 2:
            results.append((rid, 0, -1)); continue
        sigs = []

        # Signal A: ends with period
        ends_period = [t.rstrip().endswith('.') for t in texts]
        if all(ends_period) and len(texts) >= 3:
            sigs.append(0.60)
        elif any(ends_period):
            sigs.append(0.35)
        else:
            sigs.append(0.15)

        # Signal B: sentence count uniformity
        scounts = []
        for text in texts:
            sents = [s for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
            scounts.append(len(sents))
        if len(scounts) >= 2 and np.mean(scounts) > 0:
            cv = np.std(scounts) / np.mean(scounts)
            sigs.append(max(0.0, 0.7 - cv * 1.5))

        # Signal C: all start with capital
        if all(bool(re.match(r'^[A-Z]', t.strip())) for t in texts):
            sigs.append(0.45)
        else:
            sigs.append(0.15)

        if not sigs:
            results.append((rid, 0, -1)); continue
        score  = np.mean(sigs)
        is_bot = score > 0.45
        conf   = round(min(abs(score - 0.45) * 2.5, 0.80), 4)
        results.append((rid, *_vote(is_bot, conf)))
    return results


# ===========================================================================
# Ensemble aggregation
# ===========================================================================

def aggregate_ensemble(
    classifier_outputs: dict,
    weights: dict,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Signed weighted sum:
      raw_score_i  = Σ_c  w_c × vote_c × conf_c           (vote ∈ {-1, +1})
      norm_score_i = raw_score / Σ_c  w_c × |conf_c|       (range [-1, +1])
      final_vote   = +1 (bot) if norm_score > 0 else -1 (human)

    Because votes are ±1, a confident human vote actively CANCELS a weak bot vote.
    Abstentions (conf=-1) contribute nothing to either numerator or denominator.
    """
    all_ids = list(df.index)
    raw     = {rid: 0.0 for rid in all_ids}
    w_abs   = {rid: 0.0 for rid in all_ids}
    detail  = {rid: {} for rid in all_ids}

    for clf_name, output in classifier_outputs.items():
        w = weights.get(clf_name, 0.0)
        for (rid, vote, conf) in output:
            if rid not in raw:
                continue
            if conf == -1:
                detail[rid][clf_name] = {"vote": int(vote), "conf": None, "contribution": 0}
                continue
            contribution  = w * vote * conf
            raw[rid]     += contribution
            w_abs[rid]   += w * abs(conf)
            detail[rid][clf_name] = {
                "vote":         int(vote),
                "conf":         conf,
                "contribution": round(contribution, 6),
            }

    rows = []
    for rid in all_ids:
        ew    = w_abs[rid]
        norm  = raw[rid] / ew if ew > 0 else 0.0
        final = 1 if norm > 0 else -1
        rows.append({
            "ResponseId":       rid,
            "bot_score":        round(norm, 4),
            "final_vote":       final,
            "vote_label":       "bot" if final == 1 else "human",
            "raw_score":        round(raw[rid], 6),
            "effective_weight": round(ew, 4),
            "classifier_detail": json.dumps(detail[rid]),
        })
    return pd.DataFrame(rows).set_index("ResponseId")


# ===========================================================================
# Weight bootstrapping
# ===========================================================================

def bootstrap_weights(
    classifier_outputs: dict,
    weights: dict,
    assured_bots: pd.Index,
) -> dict:
    """
    Classifiers that correctly return vote=+1 for assured bots earn a higher weight.
    adjustment = 0.5 + accuracy  (range [0.5, 1.5])
    Weights are re-normalized to sum to 1 after adjustment.
    """
    if len(assured_bots) == 0:
        return weights
    updated = dict(weights)
    for clf_name, output in classifier_outputs.items():
        correct, total = 0, 0
        for (rid, vote, conf) in output:
            if rid in assured_bots and conf != -1:
                total += 1
                if vote == 1: correct += 1
        if total == 0: continue
        accuracy = correct / total
        updated[clf_name] = round(weights.get(clf_name, 0.01) * (0.5 + accuracy), 6)
    total_w = sum(updated.values())
    if total_w > 0:
        updated = {k: round(v / total_w, 6) for k, v in updated.items()}
    return updated


# ===========================================================================
# Reporting
# ===========================================================================

def print_summary(results_df: pd.DataFrame, assured_bots: pd.Index):
    total   = len(results_df)
    flagged = (results_df["final_vote"] == 1).sum()
    assured = len(assured_bots)

    print("\n" + "=" * 66)
    print("   BOT DETECTION SUMMARY")
    print("=" * 66)
    print(f"  Total respondents         : {total}")
    print(f"  Assured bots (too fast)   : {assured}")
    print(f"  Flagged as bots (+1)      : {int(flagged)}")
    print(f"  Flagged rate              : {flagged/total*100:.1f}%")
    print(f"  Vote scale                : -1 = human | +1 = bot")
    print("=" * 66)

    print("\nTop 20 highest bot-score respondents:")
    top = results_df.sort_values("bot_score", ascending=False).head(20)
    for rid, row in top.iterrows():
        marker = " ⚠ ASSURED" if rid in assured_bots else ""
        label  = "BOT  " if row["final_vote"] == 1 else "human"
        print(f"  {str(rid):30s}  score={row['bot_score']:+.4f}  [{label}]{marker}")

    print("\nPer-classifier bot-vote rates:")
    first_d = results_df["classifier_detail"].iloc[0]
    clf_names = list(json.loads(first_d).keys()) if isinstance(first_d, str) else []
    for clf_name in clf_names:
        bv, tv = 0, 0
        for ds in results_df["classifier_detail"]:
            d = json.loads(ds) if isinstance(ds, str) else ds
            if clf_name in d and d[clf_name]["conf"] is not None:
                tv += 1
                if d[clf_name]["vote"] == 1: bv += 1
        if tv:
            print(f"  {clf_name:38s}: {bv:4d}/{tv}  ({bv/tv*100:.1f}%)")
    print()


# ===========================================================================
# Main pipeline
# ===========================================================================

def run(csv_path: str, output_path: str = None, verbose: bool = True):
    print(f"Loading: {csv_path}")
    df = load_data(csv_path)
    print(f"  {len(df)} respondents, {len(df.columns)} columns")

    assured_bots = identify_assured_bots(df)
    print(f"  Assured bots (< {IMPOSSIBLE_TIME_THRESHOLD}s): {len(assured_bots)}")

    print("\nRunning classifiers...")
    classifier_outputs = {
        "no_keystrokes":         clf_no_keystrokes(df),
        "no_click_counts":       clf_no_click_counts(df),
        "recaptcha":             clf_recaptcha(df),
        "automation_flags":      clf_automation_flags(df),
        "uniform_timing":        clf_uniform_timing(df),
        "big_paste":             clf_big_paste(df),
        "low_keys_high_text":    clf_low_keys_high_text(df),
        "random_forest_outlier": clf_random_forest_outlier(df, assured_bots),
        "generic_tags":          clf_generic_tags(df),
        "text_burstiness":       clf_text_burstiness(df),
        "text_perplexity_proxy": clf_text_perplexity_proxy(df),
        "text_repetition":       clf_text_repetition(df),
        "text_ai_vocabulary":    clf_text_ai_vocabulary(df),
        "text_coherence":        clf_text_coherence(df),
        "text_structure":        clf_text_structure(df),
    }

    weights = dict(WEIGHTS)
    for name in classifier_outputs:
        if name not in weights:
            weights[name] = 0.01
    total_w = sum(weights.values())
    weights = {k: round(v / total_w, 6) for k, v in weights.items()}

    for name, output in classifier_outputs.items():
        print(f"  {name:38s}: {len(output):4d} results")

    weights = bootstrap_weights(classifier_outputs, weights, assured_bots)
    print("\nBootstrapped weights:")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {name:38s}: {w:.4f}")

    results_df = aggregate_ensemble(classifier_outputs, weights, df)

    if verbose:
        print_summary(results_df, assured_bots)

    if output_path is None:
        output_path = str(Path(csv_path).parent / "bot_detection_results.csv")

    results_df.drop(columns=["classifier_detail"]).to_csv(output_path)
    print(f"Results CSV  : {output_path}")

    detail_path = output_path.replace(".csv", "_detail.json")
    records = {}
    for rid, row in results_df.iterrows():
        records[rid] = {
            "bot_score":        row["bot_score"],
            "final_vote":       int(row["final_vote"]),
            "vote_label":       row["vote_label"],
            "raw_score":        row["raw_score"],
            "effective_weight": row["effective_weight"],
            "classifiers":      json.loads(row["classifier_detail"])
                                if isinstance(row["classifier_detail"], str)
                                else row["classifier_detail"],
        }
    with open(detail_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Detailed JSON: {detail_path}")
    return results_df


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qualtrics Bot Detection — Ensemble Classifier")
    parser.add_argument("csv",            help="Qualtrics export CSV path")
    parser.add_argument("--output", "-o", default=None, help="Output CSV path")
    parser.add_argument("--quiet",  "-q", action="store_true")
    args = parser.parse_args()
    results = run(args.csv, output_path=args.output, verbose=not args.quiet)
    n_bots  = (results["final_vote"] == 1).sum()
    print(f"\nFinal: {int(n_bots)} / {len(results)} respondents flagged as bots.")
    sys.exit(0)