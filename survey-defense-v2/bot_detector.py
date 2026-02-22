"""
==============================================================================
BOT DETECTION SYSTEM FOR SURVEY RESPONSES  v2
==============================================================================

"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import zscore, iqr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_LGB = False


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class BotDetectionResult:
    ResponseId: str
    bot_probability: float
    final_label: int        # 1=bot, -1=human
    confidence: float
    subscores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ResponseId": self.ResponseId,
            "bot_probability": round(self.bot_probability, 4),
            "final_label": self.final_label,
            "confidence": round(self.confidence, 4),
            "subscores": {k: round(v, 4) for k, v in self.subscores.items()},
        }


# ==============================================================================
# LAYER 1: FEATURE ENGINEERING
# ==============================================================================

class FeatureEngineer:
    """
    Transforms raw survey fields into a normalized feature matrix.

    v2 additions vs v1:
    - n_fast_pages: pages submitted in <2 seconds (strongest timing signal)
    - ip_subnet_count: number of responses from same /24 subnet (detects bot farms)
    - ip_is_clustered_range: flag for IP ranges with high duplication rates
    - attn_fail_count: number of failed attention/trap checks
    - duration_pct: where this response falls in the duration distribution
    - page_submit_cv: coefficient of variation of page submit times (uniformity)
    """

    LEAKAGE_COLUMNS = {
        "Q_RecaptchaScore", "Q_RelevantIDDuplicate", "Q_RelevantIDDuplicateScore",
        "Q_RelevantIDFraudScore", "Q_DuplicateRespondent", "Q_BallotBoxStuffing",
        "Q_TerminateFlag", "captcha_pass", "auto_score",
    }

    def __init__(self):
        self._duration_stats: Optional[Tuple[float, float]] = None
        self._duration_percentiles: Optional[np.ndarray] = None
        self._ip_subnet_counts: Optional[Dict[str, int]] = None
        self._ip_counts: Optional[Dict[str, int]] = None
        self._fitted = False

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize raw column quirks to consistent numeric types.
        Always call this before fit() or transform().
        """
        df = df.copy()

        # captcha_pass: "Yes"/NaN → 1.0/NaN
        if "captcha_pass" in df.columns:
            df["captcha_pass"] = df["captcha_pass"].map(
                lambda x: 1.0 if str(x).strip().lower() in ("yes", "1", "true") else np.nan
            )

        # Q_DuplicateRespondent: True/False/NaN → 1.0/0.0/NaN
        if "Q_DuplicateRespondent" in df.columns:
            def _bool_to_float(x):
                if pd.isna(x):
                    return np.nan
                s = str(x).strip().lower()
                if s in ("true", "1", "yes"):
                    return 1.0
                if s in ("false", "0", "no"):
                    return 0.0
                return np.nan
            df["Q_DuplicateRespondent"] = df["Q_DuplicateRespondent"].map(_bool_to_float)

        # auto_score: stored as 0/1/5 (count of signals); normalize to [0,1]
        if "auto_score" in df.columns:
            df["auto_score"] = pd.to_numeric(df["auto_score"], errors="coerce")
            df["auto_score"] = df["auto_score"].clip(0, 5) / 5.0

        # Duration: 0 = incomplete submission → NaN
        if "Duration (in seconds)" in df.columns:
            df["Duration (in seconds)"] = pd.to_numeric(
                df["Duration (in seconds)"], errors="coerce"
            ).replace(0, np.nan)

        return df

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        durations = df["Duration (in seconds)"].dropna()
        durations = durations[durations > 0]
        self._duration_stats = (
            float(np.median(durations)),
            float(iqr(durations) + 1e-9),
        )
        self._duration_percentiles = np.percentile(durations, np.arange(0, 101, 5))

        if "IPAddress" in df.columns:
            self._ip_counts = df["IPAddress"].value_counts().to_dict()
            # /24 subnet counts (first 3 octets)
            subnets = df["IPAddress"].apply(
                lambda x: ".".join(str(x).split(".")[:3]) if pd.notna(x) else "unknown"
            )
            self._ip_subnet_counts = subnets.value_counts().to_dict()
        else:
            self._ip_counts = {}
            self._ip_subnet_counts = {}

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "Must call fit() before transform()"
        rows = []
        for _, row in df.iterrows():
            rows.append(self._engineer_row(row))
        features = pd.DataFrame(rows, index=df.index)
        features = features.fillna(features.median(numeric_only=True))
        return features

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _engineer_row(self, row: pd.Series) -> dict:
        feats = {}

        # ── TIMING ────────────────────────────────────────────────────────────
        dur = _safe(row, "Duration (in seconds)", default=np.nan)
        total_dur = _safe(row, "Q_TotalDuration", default=dur if not np.isnan(dur) else 0.0)
        med, iqr_val = self._duration_stats

        feats["duration_zscore"] = (dur - med) / iqr_val if not np.isnan(dur) else np.nan
        feats["duration_seconds"] = np.log1p(dur) if not np.isnan(dur) else np.nan
        feats["duration_ratio"] = dur / (total_dur + 1e-9) if not np.isnan(dur) else np.nan

        # Duration percentile: where does this response sit in the full distribution?
        if not np.isnan(dur):
            feats["duration_pct"] = float(np.searchsorted(self._duration_percentiles, dur)) / 20.0
        else:
            feats["duration_pct"] = np.nan

        # ── FAST PAGE DETECTION ───────────────────────────────────────────────
        # Number of pages submitted in < 2 seconds — strongest bot timing signal
        # A human cannot meaningfully read and respond to a question in <2 seconds
        submit_cols = [c for c in row.index if "_Page Submit" in c]
        submit_vals = [_safe(row, c, np.nan) for c in submit_cols]
        submit_vals_valid = [v for v in submit_vals if not np.isnan(v)]

        n_fast = sum(1 for v in submit_vals_valid if v < 2.0)
        feats["n_fast_pages"] = float(n_fast)
        feats["fast_page_ratio"] = n_fast / (len(submit_vals_valid) + 1e-9)

        # Page submit time uniformity: bot click-through is more uniform
        if len(submit_vals_valid) >= 3:
            sv = np.array(submit_vals_valid)
            feats["page_submit_cv"] = float(np.std(sv) / (np.mean(sv) + 1e-9))
            feats["page_submit_mean"] = float(np.mean(sv))
        else:
            feats["page_submit_cv"] = np.nan
            feats["page_submit_mean"] = np.nan

        # ── CLICK PATTERNS ────────────────────────────────────────────────────
        click_cols = [c for c in row.index if "_Click Count" in c]
        click_vals = [_safe(row, c, 0.0) for c in click_cols]
        if click_vals:
            feats["click_count_mean"] = float(np.mean(click_vals))
            feats["click_count_std"] = float(np.std(click_vals))
            feats["click_entropy"] = _entropy(np.array(click_vals) + 1e-9)
        else:
            feats["click_count_mean"] = 0.0
            feats["click_count_std"] = 0.0
            feats["click_entropy"] = 0.0

        fc_cols = [c for c in row.index if "_First Click" in c]
        lc_cols = [c for c in row.index if "_Last Click" in c]
        fc_vals = [_safe(row, c, np.nan) for c in fc_cols]
        lc_vals = [_safe(row, c, np.nan) for c in lc_cols]
        dwell_times = [lc - fc for fc, lc in zip(fc_vals, lc_vals)
                       if not np.isnan(fc) and not np.isnan(lc)]
        if dwell_times:
            feats["dwell_time_mean"] = float(np.mean(dwell_times))
            feats["dwell_uniformity"] = 1.0 - min(
                float(np.std(dwell_times)) / (float(np.mean(dwell_times)) + 1e-9), 1.0
            )
        else:
            feats["dwell_time_mean"] = 0.0
            feats["dwell_uniformity"] = 0.5

        # ── KEYSTROKE FEATURES ────────────────────────────────────────────────
        ht_prefix_map = {
            "keydown": "ht_keydown_",
            "backspace": "ht_backspace_",
            "paste": "ht_paste_",
            "pasted_chars": "ht_pastedChars_",
            "total_chars": "ht_totalChars_",
        }
        agg = {}
        for key, prefix in ht_prefix_map.items():
            cols = [c for c in row.index if c.startswith(prefix)]
            vals = [_safe(row, c, np.nan) for c in cols]
            agg[key] = [v for v in vals if not np.isnan(v)]

        total_chars = sum(agg["total_chars"]) if agg["total_chars"] else 0
        pasted_chars = sum(agg["pasted_chars"]) if agg["pasted_chars"] else 0
        total_keys = sum(agg["keydown"]) if agg["keydown"] else 0
        total_backspace = sum(agg["backspace"]) if agg["backspace"] else 0
        total_paste = sum(agg["paste"]) if agg["paste"] else 0

        feats["paste_ratio"] = pasted_chars / (total_chars + 1e-9)
        feats["paste_event_ratio"] = total_paste / (total_keys + 1e-9)
        feats["backspace_rate"] = total_backspace / (total_keys + 1e-9)

        # Inter-key timing: pair by question ID to avoid shape mismatch
        med_prefix = "ht_interKeyMedMs_"
        iqr_prefix = "ht_interKeyIqrMs_"
        med_by_qid = {
            c[len(med_prefix):]: float(row[c])
            for c in row.index if c.startswith(med_prefix) and pd.notna(row[c])
        }
        iqr_by_qid = {
            c[len(iqr_prefix):]: float(row[c])
            for c in row.index if c.startswith(iqr_prefix) and pd.notna(row[c])
        }
        shared_qids = [q for q in med_by_qid if q in iqr_by_qid]
        if shared_qids:
            inter_key_meds = np.array([med_by_qid[q] for q in shared_qids])
            inter_key_iqrs = np.array([iqr_by_qid[q] for q in shared_qids])
            feats["inter_key_cv"] = float(np.mean(inter_key_iqrs / (inter_key_meds + 1e-9)))
            feats["inter_key_median_ms"] = float(np.median(inter_key_meds))
        elif med_by_qid:
            feats["inter_key_cv"] = np.nan
            feats["inter_key_median_ms"] = float(np.median(list(med_by_qid.values())))
        else:
            feats["inter_key_cv"] = np.nan
            feats["inter_key_median_ms"] = np.nan

        # HT flags: fraction of questions triggering each flag
        for fp in ["ht_flag_bigPaste_", "ht_flag_lowKeysHighText_", "ht_flag_uniformTiming_"]:
            fcols = [c for c in row.index if c.startswith(fp)]
            fvals = [_safe(row, c, 0.0) for c in fcols]
            feat_name = fp.replace("ht_flag_", "").rstrip("_")
            feats[feat_name] = float(np.mean(fvals)) if fvals else 0.0

        # ── AUTOMATION FINGERPRINT ────────────────────────────────────────────
        feats["auto_webdriver"] = float(_safe(row, "auto_webdriver", 0.0))
        feats["auto_phantom"] = float(_safe(row, "auto_phantom", 0.0))
        feats["auto_nightmare"] = float(_safe(row, "auto_nightmare", 0.0))
        feats["auto_domAutomation"] = float(_safe(row, "auto_domAutomation", 0.0))
        plugins = _safe(row, "auto_pluginsLen", np.nan)
        feats["no_plugins"] = float(1.0 if (not np.isnan(plugins) and plugins == 0) else 0.0)
        feats["plugins_len"] = float(plugins) if not np.isnan(plugins) else 2.0
        langs = _safe(row, "auto_langsLen", np.nan)
        feats["no_langs"] = float(1.0 if (not np.isnan(langs) and langs == 0) else 0.0)

        feats["automation_composite"] = (
            feats["auto_webdriver"] * 0.4
            + feats["auto_phantom"] * 0.3
            + feats["auto_nightmare"] * 0.1
            + feats["auto_domAutomation"] * 0.1
            + feats["no_plugins"] * 0.05
            + feats["no_langs"] * 0.05
        )

        # ── MOTOR ANALYSIS ────────────────────────────────────────────────────
        motor_score = _safe(row, "MotorCheckScore", np.nan)
        motor_smooth = _safe(row, "MotorCheckSmoothness", np.nan)
        motor_pts = _safe(row, "MotorCheckPoints", np.nan)
        motor_strokes = _safe(row, "MotorCheckStrokeCount", np.nan)
        motor_speed = _safe(row, "MotorCheckAvgSpeed", np.nan)
        motor_bins = _safe(row, "MotorCheckVisitedBins", np.nan)

        feats["motor_score"] = float(motor_score) if not np.isnan(motor_score) else 0.5
        feats["motor_smoothness"] = float(motor_smooth) if not np.isnan(motor_smooth) else 0.5
        feats["motor_pts_per_stroke"] = (
            float(motor_pts / (motor_strokes + 1e-9))
            if not np.isnan(motor_pts) and not np.isnan(motor_strokes) else 0.0
        )
        feats["motor_speed"] = float(motor_speed) if not np.isnan(motor_speed) else 0.0
        feats["motor_visited_bins"] = float(motor_bins) if not np.isnan(motor_bins) else 0.0

        # ── IP / NETWORK FEATURES ─────────────────────────────────────────────
        ip = str(row.get("IPAddress", ""))
        ip_parts = ip.split(".")
        subnet = ".".join(ip_parts[:3]) if len(ip_parts) >= 3 else "unknown"
        ip_first2 = ".".join(ip_parts[:2]) if len(ip_parts) >= 2 else "unknown"

        ip_count = self._ip_counts.get(ip, 1)
        feats["ip_frequency"] = np.log1p(ip_count)

        # Subnet clustering: bots from same farm share /24 subnets
        subnet_count = self._ip_subnet_counts.get(subnet, 1)
        feats["ip_subnet_count"] = float(subnet_count)
        feats["ip_subnet_log"] = np.log1p(subnet_count)

        # Known suspicious IP ranges from this dataset (35.x = clustered datacenter)
        # 35.1, 35.2, 35.3 — extremely high duplication rates in THIS survey
        feats["ip_in_suspect_range"] = float(
            ip_first2 in ("35.1", "35.2", "35.3", "35.0")
            or ip.startswith("35.") and len(ip_parts) >= 2 and ip_parts[1] in ("1","2","3","0")
        )

        # ── ATTENTION / TRAP CHECKS ───────────────────────────────────────────
        # attn1: multiple-choice attention check
        attn1_val = str(row.get("attn1", "")).strip()
        # attn2: type "YEVRUS" (survey backward) — tests if respondent reads instructions
        attn2_val = str(row.get("attn2", "")).strip().upper()
        attn2_correct = attn2_val in ("YEVRUS",)
        feats["attn1_correct"] = float(attn1_val != "" and attn1_val != "nan")  # non-null = attempted
        feats["attn2_correct"] = float(attn2_correct)

        # ── METADATA ──────────────────────────────────────────────────────────
        tz = str(row.get("BrowserTimeZone", ""))
        lang = str(row.get("ComputerLanguage", ""))
        feats["has_timezone"] = float(1.0 if tz and tz not in ("nan", "UTC") else 0.0)
        feats["has_language"] = float(1.0 if lang and lang != "nan" else 0.0)

        return feats


# ==============================================================================
# LAYER 2: HEURISTIC RISK LAYER
# ==============================================================================

class HeuristicRiskLayer:
    """
    Produces interpretable subscores and pseudo-labels.

    v2 key changes:
    - duplication_risk: Q_DuplicateRespondent now anchors the score at 0.75
      when set (it's the strongest vendor signal for this dataset)
    - behavioral_risk: n_fast_pages is now a primary component
    - ip_risk: new subscore for network-level clustering
    - pseudo-labeling: uses Q_DuplicateRespondent as a strong anchor
    """

    _duration_p10: Optional[float] = None
    _duration_p90: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> "HeuristicRiskLayer":
        durations = pd.to_numeric(df["Duration (in seconds)"], errors="coerce")
        durations = durations[(durations > 0) & durations.notna()]
        self._duration_p10 = float(np.percentile(durations, 10)) if len(durations) else 120.0
        self._duration_p90 = float(np.percentile(durations, 90)) if len(durations) else 1200.0
        return self

    def compute(self, row: pd.Series, features: Optional[pd.Series] = None) -> Dict[str, float]:
        return {
            "automation_risk": self._automation_risk(row),
            "behavioral_risk": self._behavioral_risk(row, features),
            "duplication_risk": self._duplication_risk(row),
        }

    def compute_batch(self, df: pd.DataFrame, features_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        results = []
        for idx, row in df.iterrows():
            feat_row = features_df.loc[idx] if features_df is not None else None
            results.append(self.compute(row, feat_row))
        return pd.DataFrame(results, index=df.index)

    def _automation_risk(self, row: pd.Series) -> float:
        risk = 0.0
        weights = 0.0
        hard_hits = 0
        for col, weight in [
            ("auto_webdriver", 0.35), ("auto_phantom", 0.25),
            ("auto_nightmare", 0.15), ("auto_domAutomation", 0.15),
        ]:
            val = _safe(row, col, np.nan)
            if not np.isnan(val):
                risk += weight * float(val)
                weights += weight
                if float(val) == 1.0:
                    hard_hits += 1

        if hard_hits > 0:
            return max(risk / (weights + 1e-9), 0.85)

        plugins = _safe(row, "auto_pluginsLen", np.nan)
        if not np.isnan(plugins):
            risk += 0.05 * float(plugins == 0)
            weights += 0.05

        auto_s = _safe(row, "auto_score", np.nan)
        if not np.isnan(auto_s):
            risk += 0.05 * float(auto_s)
            weights += 0.05

        return risk / (weights + 1e-9) if weights > 0 else 0.0

    def _behavioral_risk(self, row: pd.Series, features: Optional[pd.Series] = None) -> float:
        risks = []

        # Fast page count: strong primary signal
        if features is not None and "n_fast_pages" in features and not np.isnan(features["n_fast_pages"]):
            n_fast = features["n_fast_pages"]
            # ≥8 fast pages → high risk; ≥5 → moderate risk
            risks.append(float(np.clip(n_fast / 15.0, 0, 1)))

        # Duration: short completions are suspicious
        dur = _safe(row, "Duration (in seconds)", np.nan)
        if not np.isnan(dur) and self._duration_p10 is not None:
            if dur < self._duration_p10:
                risks.append(min(1.0, self._duration_p10 / (dur + 1e-9) * 0.15))
            elif dur > self._duration_p90 * 1.5:
                risks.append(0.1)  # suspiciously slow is a weak signal

        # Paste signals
        total_chars = _sum_prefix(row, "ht_totalChars_")
        pasted_chars = _sum_prefix(row, "ht_pastedChars_")
        if total_chars > 0:
            paste_ratio = pasted_chars / total_chars
            risks.append(expit(10 * (paste_ratio - 0.6)))

        # Keystroke flags
        uniform = _mean_flag(row, "ht_flag_uniformTiming_")
        if uniform is not None:
            risks.append(uniform)

        lkht = _mean_flag(row, "ht_flag_lowKeysHighText_")
        if lkht is not None:
            risks.append(lkht)

        bp = _mean_flag(row, "ht_flag_bigPaste_")
        if bp is not None:
            risks.append(bp)

        # No backspaces in long text = suspicious
        total_keys = _sum_prefix(row, "ht_keydown_")
        total_backspace = _sum_prefix(row, "ht_backspace_")
        if total_keys > 50 and total_backspace == 0:
            risks.append(0.7)

        # Attention check failure
        attn2_val = str(row.get("attn2", "")).strip().upper()
        if attn2_val and attn2_val != "NAN":
            if attn2_val not in ("YEVRUS",):
                risks.append(0.6)  # Failed explicit instruction-following task

        # Motor signals
        motor_score = _safe(row, "MotorCheckScore", np.nan)
        if not np.isnan(motor_score):
            risks.append(1.0 - float(np.clip(motor_score, 0, 1)))

        motor_smooth = _safe(row, "MotorCheckSmoothness", np.nan)
        if not np.isnan(motor_smooth):
            risks.append(expit(8 * (float(motor_smooth) - 0.9)))

        return float(np.mean(risks)) if risks else 0.0

    def _duplication_risk(self, row: pd.Series) -> float:
        """
        v2 CHANGE: Q_DuplicateRespondent is now the primary anchor.
        When True, score is floored at 0.75 regardless of other signals.
        This field is the strongest vendor-provided signal in this dataset
        and was being diluted to ~0.05 contribution in v1.
        """
        risk = 0.0
        weights = 0.0

        # Q_DuplicateRespondent: platform-level deduplication — strongest signal
        dup = _safe(row, "Q_DuplicateRespondent", np.nan)
        if not np.isnan(dup):
            if float(dup) == 1.0:
                # Hard floor: this person has a known duplicate in the system
                return 0.80
            weights += 0.40  # Present and False = strong human signal

        # Ballot box stuffing
        bbs = _safe(row, "Q_BallotBoxStuffing", np.nan)
        if not np.isnan(bbs):
            risk += 0.35 * float(bbs)
            weights += 0.35

        # RelevantID duplicate score
        dup_score = _safe(row, "Q_RelevantIDDuplicateScore", np.nan)
        if not np.isnan(dup_score):
            risk += 0.20 * float(np.clip(dup_score, 0, 1))
            weights += 0.20

        # Captcha failure
        captcha = _safe(row, "captcha_pass", np.nan)
        if not np.isnan(captcha):
            risk += 0.05 * (1.0 - float(captcha))
            weights += 0.05

        return risk / (weights + 1e-9) if weights > 0 else 0.0

    def generate_pseudo_labels(
        self, df: pd.DataFrame, subscores: pd.DataFrame, features_df: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        v2 REDESIGN: Uses signals that these bots actually fail on.

        Strong bot anchors for THIS dataset:
        - Q_DuplicateRespondent = True (50% of 35.x bots have this)
        - IP in suspect range (35.x) COMBINED WITH short duration or dup flag
        - n_fast_pages >= 8 (bot click-through behavior)

        Strong human anchors:
        - Not in suspect IP range
        - Not duplicate
        - Duration in mid-range (60–900s)
        - Good recaptcha + captcha

        We intentionally label bots that DON'T have recaptcha/automation failures
        because those are the primary evasion pattern in this dataset.
        """
        labels = pd.Series(0, index=df.index, dtype=int)

        for idx, row in df.iterrows():
            ss = subscores.loc[idx]
            auto_risk = ss["automation_risk"]
            behav_risk = ss["behavioral_risk"]
            dup_risk = ss["duplication_risk"]

            feat_row = features_df.loc[idx] if features_df is not None else None

            dur = _safe(row, "Duration (in seconds)", np.nan)
            recaptcha = _safe(row, "Q_RecaptchaScore", np.nan)
            captcha = _safe(row, "captcha_pass", np.nan)
            fraud_score = _safe(row, "Q_RelevantIDFraudScore", np.nan)
            bbs = _safe(row, "Q_BallotBoxStuffing", 0.0)
            dup_respondent = _safe(row, "Q_DuplicateRespondent", 0.0)
            raw_webdriver = _safe(row, "auto_webdriver", 0.0)
            raw_phantom = _safe(row, "auto_phantom", 0.0)
            raw_nightmare = _safe(row, "auto_nightmare", 0.0)

            # Incomplete rows (no duration) → uncertain
            if np.isnan(dur):
                labels[idx] = 0
                continue

            # Feature-derived signals
            n_fast = float(feat_row["n_fast_pages"]) if feat_row is not None and "n_fast_pages" in feat_row else 0.0
            ip_suspect = float(feat_row["ip_in_suspect_range"]) if feat_row is not None else 0.0
            subnet_count = float(feat_row["ip_subnet_count"]) if feat_row is not None else 1.0

            attn2_val = str(row.get("attn2", "")).strip().upper()
            attn2_failed = attn2_val and attn2_val != "NAN" and attn2_val not in ("YEVRUS",)

            # ── HARD OVERRIDES ──
            if raw_webdriver == 1.0 or raw_phantom == 1.0 or raw_nightmare == 1.0:
                labels[idx] = 1
                continue

            if bbs == 1.0:
                labels[idx] = 1
                continue

            # ── BOT SIGNAL ACCUMULATION ──
            bot_signals = 0

            # Duplication signals (strongest in this dataset)
            if dup_respondent == 1.0:
                bot_signals += 3  # Platform-level duplicate detection
            if dup_risk > 0.6:
                bot_signals += 1

            # Automation fingerprinting
            if auto_risk > 0.7:
                bot_signals += 2
            elif auto_risk > 0.35:
                bot_signals += 1

            # Behavioral anomalies
            if behav_risk > 0.65:
                bot_signals += 2
            elif behav_risk > 0.4:
                bot_signals += 1

            # Network clustering (35.x range = high bot density)
            if ip_suspect == 1.0:
                bot_signals += 2
            elif subnet_count >= 3:
                bot_signals += 1

            # Fast-click behavior
            if n_fast >= 8:
                bot_signals += 2
            elif n_fast >= 5:
                bot_signals += 1

            # Recaptcha (weak signal here — bots pass it)
            if not np.isnan(recaptcha) and recaptcha < 0.3:
                bot_signals += 2
            elif not np.isnan(recaptcha) and recaptcha < 0.5:
                bot_signals += 1

            # Captcha absence
            if np.isnan(captcha):
                bot_signals += 1

            # Short duration
            if not np.isnan(dur) and self._duration_p10 and dur < self._duration_p10:
                bot_signals += 1

            # ── HUMAN SIGNAL ACCUMULATION ──
            human_signals = 0

            if dup_respondent == 0.0 and not np.isnan(_safe(row, "Q_DuplicateRespondent", np.nan)):
                human_signals += 2
            if auto_risk < 0.1:
                human_signals += 2
            if behav_risk < 0.2:
                human_signals += 2
            if not np.isnan(recaptcha) and recaptcha > 0.8:
                human_signals += 1  # Weak — bots also pass this
            if not np.isnan(captcha) and captcha == 1.0:
                human_signals += 1
            if not np.isnan(dur) and 120 < dur < 1200:
                human_signals += 1
            if ip_suspect == 0.0 and subnet_count <= 1:
                human_signals += 2
            if n_fast <= 2:
                human_signals += 1

            # Assign labels with asymmetric thresholds
            if bot_signals >= 3 and bot_signals > human_signals:
                labels[idx] = 1
            elif human_signals >= 6 and bot_signals <= 1:
                labels[idx] = -1
            else:
                labels[idx] = 0  # uncertain

        return labels


# ==============================================================================
# LAYER 3: UNSUPERVISED ANOMALY DETECTION
# ==============================================================================

class AnomalyDetector:
    """
    Isolation Forest on behavioral + network features.
    v2: includes n_fast_pages and ip_subnet_count in feature set.
    """

    ANOMALY_FEATURES = [
        "duration_zscore", "duration_ratio", "duration_pct",
        "n_fast_pages", "fast_page_ratio", "page_submit_cv",
        "click_entropy", "click_count_std", "dwell_uniformity",
        "paste_ratio", "paste_event_ratio", "backspace_rate",
        "inter_key_cv", "inter_key_median_ms",
        "uniformTiming", "lowKeysHighText", "bigPaste",
        "motor_score", "motor_smoothness",
        "ip_subnet_log", "ip_in_suspect_range",
        "attn2_correct",
    ]

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            max_samples="auto",
            bootstrap=False,
            n_jobs=-1,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, features: pd.DataFrame) -> "AnomalyDetector":
        X = self._select_features(features)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._fitted = True
        return self

    def anomaly_scores(self, features: pd.DataFrame) -> np.ndarray:
        assert self._fitted
        X = self._select_features(features)
        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        return expit(-raw_scores * 5.0)

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.ANOMALY_FEATURES if c in features.columns]
        X = features[available].copy()
        X = X.fillna(X.median(numeric_only=True))
        return X


# ==============================================================================
# LAYER 4: SUPERVISED CLASSIFIER
# ==============================================================================

class SupervisedClassifier:
    """
    Gradient boosting on pseudo-labeled data.
    v2: includes n_fast_pages, ip_subnet features, attn2_correct.
    Excludes Q_DuplicateRespondent (leakage — it generates pseudo-labels).
    """

    SUPERVISED_FEATURES = [
        "duration_zscore", "duration_ratio", "duration_pct", "duration_seconds",
        "n_fast_pages", "fast_page_ratio", "page_submit_cv", "page_submit_mean",
        "click_entropy", "click_count_mean", "click_count_std",
        "dwell_uniformity", "dwell_time_mean",
        "paste_ratio", "paste_event_ratio", "backspace_rate",
        "inter_key_cv", "inter_key_median_ms",
        "uniformTiming", "lowKeysHighText", "bigPaste",
        "auto_webdriver", "auto_phantom", "auto_nightmare", "auto_domAutomation",
        "no_plugins", "plugins_len", "no_langs",
        "automation_composite",
        "motor_score", "motor_smoothness", "motor_pts_per_stroke",
        "motor_speed", "motor_visited_bins",
        "ip_frequency", "ip_subnet_log", "ip_in_suspect_range",
        "has_timezone", "has_language",
        "attn2_correct",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = RobustScaler()
        self._fitted = False
        self.model = None
        self.feature_importances_ = None

    def fit(self, features: pd.DataFrame, pseudo_labels: pd.Series) -> "SupervisedClassifier":
        mask = pseudo_labels != 0
        n_confident = mask.sum()
        n_bot = int((pseudo_labels[mask] == 1).sum())
        n_human = int(n_confident - n_bot)

        if n_confident < 10 or n_bot < 2:
            print(f"      SKIPPING: {n_bot} bot pseudo-labels (need ≥2)")
            self._fitted = False
            return self

        if n_bot < 20:
            print(f"      WARNING: only {n_bot} bot pseudo-labels — using lightweight model")

        X = self._select_features(features[mask])
        y = (pseudo_labels[mask] == 1).astype(int)
        X_scaled = self.scaler.fit_transform(X)

        scale_pos = n_human / (n_bot + 1e-9)
        n_estimators = 100 if n_bot < 20 else 400
        max_depth = 4 if n_bot < 20 else 6

        if HAS_LGB:
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators, learning_rate=0.05, max_depth=max_depth,
                num_leaves=31, min_child_samples=max(3, n_bot // 4),
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
                n_jobs=-1, random_state=self.random_state, verbose=-1,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators, learning_rate=0.05, max_depth=max_depth,
                min_samples_leaf=max(3, n_bot // 4), subsample=0.8,
                random_state=self.random_state,
            )

        self.model.fit(X_scaled, y)

        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = pd.Series(
                self.model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)

        self._fitted = True
        print(f"      Trained on {n_confident} samples ({n_bot} bots, {n_human} humans)")
        return self

    def predict_proba(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        if not self._fitted:
            return None
        X = self._select_features(features)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.SUPERVISED_FEATURES if c in features.columns]
        X = features[available].copy()
        return X.fillna(X.median(numeric_only=True))


# ==============================================================================
# LAYER 5: META-CLASSIFIER
# ==============================================================================

class MetaClassifier:
    """
    Log-odds aggregation with fallback weighted combination.

    v2 fallback weights reflect THIS dataset's signal quality:
    - duplication_risk: 0.40 (platform deduplication = strongest signal here)
    - behavioral_risk:  0.30 (fast pages + timing)
    - automation_risk:  0.15 (almost never fires, but definitive when it does)
    - anomaly_risk:     0.15 (unsupervised baseline)
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.meta_model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
        self.calibrated_model = None
        self._fitted = False
        self._single_class = None

    def fit(self, automation_risk, behavioral_risk, duplication_risk, anomaly_risk,
            supervised_prob, pseudo_labels: pd.Series) -> "MetaClassifier":
        mask = pseudo_labels != 0
        y = (pseudo_labels[mask] == 1).astype(int).values
        n_bot = y.sum()
        n_human = len(y) - n_bot

        if len(np.unique(y)) < 2 or n_bot < 5:
            print(f"WARNING: Too few bot labels (bots={n_bot}) — using weighted-subscore fallback")
            self._single_class = 0
            self._fitted = True
            return self
        self._single_class = None

        X_meta = self._build_meta_features(
            automation_risk[mask], behavioral_risk[mask],
            duplication_risk[mask], anomaly_risk[mask],
            supervised_prob[mask] if supervised_prob is not None else None,
        )
        n = mask.sum()
        method = "isotonic" if n > 1000 else "sigmoid"
        cv = min(5, max(2, n_bot))

        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.meta_model, method=method, cv=cv,
        )
        self.calibrated_model.fit(X_meta, y)
        self._fitted = True
        return self

    def predict(self, automation_risk, behavioral_risk, duplication_risk,
                anomaly_risk, supervised_prob) -> Tuple[np.ndarray, np.ndarray]:
        assert self._fitted

        if self._single_class is not None:
            # Weighted fallback: duplication is most reliable in this dataset
            weights = np.array([0.15, 0.30, 0.40, 0.15])
            stacked = np.column_stack([automation_risk, behavioral_risk, duplication_risk, anomaly_risk])
            bot_proba = np.clip(stacked @ weights, 0, 1)
            confidence = 2.0 * np.abs(bot_proba - 0.5)
            return bot_proba, confidence

        X_meta = self._build_meta_features(automation_risk, behavioral_risk,
                                           duplication_risk, anomaly_risk, supervised_prob)
        bot_proba = self.calibrated_model.predict_proba(X_meta)[:, 1]
        confidence = 2.0 * np.abs(bot_proba - 0.5)
        return bot_proba, confidence

    def _build_meta_features(self, auto, behav, dup, anomaly, supervised):
        def safe_logit(x):
            return logit(np.clip(x, 1e-6, 1 - 1e-6))
        cols = [safe_logit(auto), safe_logit(behav), safe_logit(dup), safe_logit(anomaly)]
        if supervised is not None:
            cols.append(safe_logit(supervised))
        return np.column_stack(cols)

    def update_threshold(self, threshold: float) -> None:
        self.threshold = threshold


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

class BotDetectionPipeline:
    def __init__(self, bot_threshold: float = 0.5):
        self.feature_engineer = FeatureEngineer()
        self.heuristic_layer = HeuristicRiskLayer()
        self.anomaly_detector = AnomalyDetector()
        self.supervised_classifier = SupervisedClassifier()
        self.meta_classifier = MetaClassifier(threshold=bot_threshold)
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "BotDetectionPipeline":
        print("=" * 60)
        print("FITTING BOT DETECTION PIPELINE  v2")
        print("=" * 60)

        df = FeatureEngineer.preprocess(df)

        print("\n[1/6] Engineering features...")
        features = self.feature_engineer.fit_transform(df)
        print(f"      Generated {features.shape[1]} features for {len(df)} responses")

        print("\n[2/6] Computing heuristic subscores...")
        self.heuristic_layer.fit(df)
        subscores = self.heuristic_layer.compute_batch(df, features)

        print("\n[3/6] Generating pseudo-labels...")
        pseudo_labels = self.heuristic_layer.generate_pseudo_labels(df, subscores, features)
        n_bot = (pseudo_labels == 1).sum()
        n_human = (pseudo_labels == -1).sum()
        n_uncertain = (pseudo_labels == 0).sum()
        print(f"      Bot: {n_bot}, Human: {n_human}, Uncertain: {n_uncertain}")
        print(f"      Label coverage: {(n_bot + n_human) / len(df):.1%}")
        if n_bot == 0:
            print("      ⚠ ZERO bot pseudo-labels — check IP/duplication signals in your data")

        print("\n[4/6] Fitting anomaly detector...")
        self.anomaly_detector.fit(features)
        anomaly_risk = self.anomaly_detector.anomaly_scores(features)

        print("\n[5/6] Training supervised classifier...")
        self.supervised_classifier.fit(features, pseudo_labels)
        supervised_prob = self.supervised_classifier.predict_proba(features)
        if supervised_prob is None:
            print("      Supervised classifier inactive — other layers carry the weight")

        print("\n[6/6] Fitting meta-classifier...")
        self.meta_classifier.fit(
            automation_risk=subscores["automation_risk"].values,
            behavioral_risk=subscores["behavioral_risk"].values,
            duplication_risk=subscores["duplication_risk"].values,
            anomaly_risk=anomaly_risk,
            supervised_prob=supervised_prob,
            pseudo_labels=pseudo_labels,
        )

        self._fitted = True
        print("\n✓ Pipeline fitted successfully")
        return self

    def predict(self, df: pd.DataFrame) -> List[BotDetectionResult]:
        assert self._fitted
        df = FeatureEngineer.preprocess(df)
        features = self.feature_engineer.transform(df)
        subscores = self.heuristic_layer.compute_batch(df, features)
        anomaly_risk = self.anomaly_detector.anomaly_scores(features)
        supervised_prob = self.supervised_classifier.predict_proba(features)

        bot_proba, confidence = self.meta_classifier.predict(
            automation_risk=subscores["automation_risk"].values,
            behavioral_risk=subscores["behavioral_risk"].values,
            duplication_risk=subscores["duplication_risk"].values,
            anomaly_risk=anomaly_risk,
            supervised_prob=supervised_prob,
        )

        results = []
        for i, (idx, row) in enumerate(df.iterrows()):
            p = float(bot_proba[i])
            label = 1 if p >= self.meta_classifier.threshold else -1
            results.append(BotDetectionResult(
                ResponseId=str(row.get("ResponseId", str(idx))),
                bot_probability=p,
                final_label=label,
                confidence=float(confidence[i]),
                subscores={
                    "automation_risk": float(subscores.loc[idx, "automation_risk"]),
                    "behavioral_risk": float(subscores.loc[idx, "behavioral_risk"]),
                    "duplication_risk": float(subscores.loc[idx, "duplication_risk"]),
                    "anomaly_risk": float(anomaly_risk[i]),
                },
            ))
        return results

    def fit_predict(self, df: pd.DataFrame) -> List[BotDetectionResult]:
        return self.fit(df).predict(df)


# ==============================================================================
# VALIDATION
# ==============================================================================

class ValidationFramework:
    @staticmethod
    def compute_agreement_metrics(results: List[BotDetectionResult]) -> dict:
        n = len(results)
        strong_bot = strong_human = contradictory = 0
        for r in results:
            ss = r.subscores
            high = sum(1 for v in ss.values() if v > 0.6)
            low = sum(1 for v in ss.values() if v < 0.3)
            if high >= 3:
                strong_bot += 1
            elif low >= 3:
                strong_human += 1
            elif high >= 1 and low >= 1:
                contradictory += 1
        return {
            "strong_bot_pct": strong_bot / n,
            "strong_human_pct": strong_human / n,
            "contradictory_pct": contradictory / n,
            "flagged_as_bot": sum(1 for r in results if r.final_label == 1) / n,
            "mean_confidence": float(np.mean([r.confidence for r in results])),
            "pct_high_confidence": sum(1 for r in results if r.confidence > 0.8) / n,
        }

    @staticmethod
    def threshold_analysis(results: List[BotDetectionResult]) -> pd.DataFrame:
        rows = []
        for t in np.arange(0.1, 1.0, 0.05):
            flagged = [r for r in results if r.bot_probability >= t]
            n_flagged = len(flagged)
            if n_flagged == 0:
                continue
            proxy_tp = sum(
                1 for r in flagged
                if sum(1 for v in r.subscores.values() if v > 0.5) >= 2
            )
            rows.append({
                "threshold": round(t, 2),
                "n_flagged": n_flagged,
                "flagged_pct": round(n_flagged / len(results), 3),
                "proxy_precision": round(proxy_tp / n_flagged, 3),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def score_drift_report(results1, results2) -> dict:
        def extract(results, key):
            return np.array([r.subscores[key] for r in results])
        keys = ["automation_risk", "behavioral_risk", "duplication_risk", "anomaly_risk"]
        report = {}
        for k in keys:
            s1, s2 = extract(results1, k), extract(results2, k)
            psi = _compute_psi(s1, s2)
            report[k] = {"mean_1": float(np.mean(s1)), "mean_2": float(np.mean(s2)),
                         "psi": float(psi), "drift_flag": psi > 0.25}
        report["bot_rate_1"] = sum(1 for r in results1 if r.final_label == 1) / len(results1)
        report["bot_rate_2"] = sum(1 for r in results2 if r.final_label == 1) / len(results2)
        return report


# ==============================================================================
# UTILITIES
# ==============================================================================

def _safe(row: pd.Series, col: str, default=np.nan) -> float:
    val = row.get(col, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def _sum_prefix(row: pd.Series, prefix: str) -> float:
    cols = [c for c in row.index if c.startswith(prefix)]
    return float(sum(_safe(row, c, 0.0) for c in cols))

def _mean_flag(row: pd.Series, prefix: str) -> Optional[float]:
    cols = [c for c in row.index if c.startswith(prefix)]
    if not cols:
        return None
    return float(np.mean([_safe(row, c, 0.0) for c in cols]))

def _entropy(probs: np.ndarray) -> float:
    probs = probs / (probs.sum() + 1e-9)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    raw = -float(np.sum(probs * np.log2(probs)))
    return raw / (np.log2(len(probs)) + 1e-9)

def _compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    bins[0] = -np.inf
    bins[-1] = np.inf
    exp_c = np.histogram(expected, bins=bins)[0] + 1
    act_c = np.histogram(actual, bins=bins)[0] + 1
    exp_p = exp_c / exp_c.sum()
    act_p = act_c / act_c.sum()
    return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))


# ==============================================================================
# SYNTHETIC DATA FOR TESTING
# ==============================================================================

def generate_synthetic_data(n: int = 500, bot_rate: float = 0.30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_bot = int(n * bot_rate)
    n_human = n - n_bot
    rows = []

    for i in range(n_human):
        dur = rng.lognormal(mean=4.5, sigma=0.7)
        row = {
            "ResponseId": f"H{i:04d}", "Duration (in seconds)": dur,
            "Q_TotalDuration": dur * rng.uniform(1.0, 1.3),
            "auto_webdriver": 0, "auto_phantom": 0, "auto_nightmare": 0,
            "auto_domAutomation": 0, "auto_pluginsLen": rng.integers(2, 8),
            "auto_langsLen": rng.integers(1, 4), "auto_score": rng.uniform(0, 1),
            "ht_keydown_QID631": rng.integers(20, 100),
            "ht_backspace_QID631": rng.integers(1, 10),
            "ht_paste_QID631": 0, "ht_pastedChars_QID631": 0,
            "ht_totalChars_QID631": rng.integers(30, 120),
            "ht_interKeyMedMs_QID631": rng.uniform(120, 250),
            "ht_interKeyIqrMs_QID631": rng.uniform(60, 150),
            "ht_flag_bigPaste_QID631": 0, "ht_flag_lowKeysHighText_QID631": 0,
            "ht_flag_uniformTiming_QID631": 0,
            "Q_RecaptchaScore": rng.uniform(0.7, 1.0),
            "Q_DuplicateRespondent": False,
            "Q_RelevantIDDuplicateScore": rng.uniform(0, 0.2),
            "Q_RelevantIDFraudScore": rng.uniform(0, 0.2),
            "Q_DuplicateRespondent": False, "Q_BallotBoxStuffing": np.nan,
            "captcha_pass": "Yes",
            "IPAddress": f"192.168.{rng.integers(1,255)}.{rng.integers(1,255)}",
            "BrowserTimeZone": "America/New_York", "ComputerLanguage": "en-US",
            "attn2": "YEVRUS",
            "consent_Page Submit": rng.uniform(5, 30),
            "gender_Page Submit": rng.uniform(2, 8),
            "ideo5_Page Submit": rng.uniform(3, 15),
            "MotorCheckScore": rng.uniform(0.6, 0.95),
            "MotorCheckSmoothness": rng.uniform(0.3, 0.7),
        }
        rows.append(row)

    for i in range(n_bot):
        dur = rng.choice([rng.uniform(100, 350), rng.uniform(350, 500)])
        use_35 = rng.random() > 0.3
        ip = f"35.1.{rng.integers(1,50)}.{rng.integers(1,250)}" if use_35 else \
             f"10.{rng.integers(0,5)}.{rng.integers(1,50)}.{rng.integers(1,250)}"
        n_fast = rng.integers(5, 20)
        row = {
            "ResponseId": f"B{i:04d}", "Duration (in seconds)": dur,
            "Q_TotalDuration": dur, "auto_webdriver": 0, "auto_phantom": 0,
            "auto_nightmare": 0, "auto_domAutomation": 0,
            "auto_pluginsLen": rng.choice([0, 0, 0, 2]),
            "auto_langsLen": rng.choice([0, 1]), "auto_score": 1,
            "ht_keydown_QID631": rng.integers(0, 5),
            "ht_backspace_QID631": 0, "ht_paste_QID631": 1,
            "ht_pastedChars_QID631": rng.integers(50, 200),
            "ht_totalChars_QID631": rng.integers(50, 200),
            "ht_interKeyMedMs_QID631": rng.uniform(50, 100),
            "ht_interKeyIqrMs_QID631": rng.uniform(0, 10),
            "ht_flag_bigPaste_QID631": 1, "ht_flag_lowKeysHighText_QID631": 1,
            "ht_flag_uniformTiming_QID631": 1,
            "Q_RecaptchaScore": rng.uniform(0.85, 1.0),  # bots pass recaptcha
            "Q_DuplicateRespondent": rng.choice([True, True, False]),
            "Q_RelevantIDDuplicateScore": rng.uniform(0.3, 0.9),
            "Q_RelevantIDFraudScore": rng.uniform(0.2, 0.8),
            "Q_BallotBoxStuffing": np.nan,
            "captcha_pass": "Yes",  # bots pass captcha
            "IPAddress": ip,
            "BrowserTimeZone": "", "ComputerLanguage": "",
            "attn2": rng.choice(["YEVRUS", "Survey", "survey"]),
            "consent_Page Submit": rng.uniform(1, 3),
            "gender_Page Submit": rng.uniform(0.5, 1.8),
            "ideo5_Page Submit": rng.uniform(0.5, 1.5),
            "MotorCheckScore": np.nan, "MotorCheckSmoothness": np.nan,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Generating synthetic data (bots that evade recaptcha/captcha)...")
    df = generate_synthetic_data(n=400, bot_rate=0.30)
    true_bot = df["ResponseId"].str.startswith("B")
    print(f"Dataset: {len(df)} responses ({true_bot.sum()} bots)\n")

    pipeline = BotDetectionPipeline(bot_threshold=0.5)
    results = pipeline.fit_predict(df)

    n_bot_detected = sum(1 for r in results if r.final_label == 1)
    ids = {r.ResponseId: r for r in results}
    tp = sum(1 for r in results if r.final_label == 1 and r.ResponseId.startswith("B"))
    fp = sum(1 for r in results if r.final_label == 1 and r.ResponseId.startswith("H"))
    fn = sum(1 for r in results if r.final_label == -1 and r.ResponseId.startswith("B"))

    print(f"\n{'='*60}")
    print(f"RESULTS  (true positives / false positives / false negatives)")
    print(f"{'='*60}")
    print(f"  Flagged as bot: {n_bot_detected} ({n_bot_detected/len(results):.1%})")
    print(f"  True positives:  {tp}  |  False positives: {fp}  |  False negatives: {fn}")
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    print(f"  Precision: {prec:.2f}  Recall: {rec:.2f}  F1: {2*prec*rec/(prec+rec+1e-9):.2f}")

    import json
    print(f"\nSample bot result:")
    bot_results = [r for r in results if r.final_label == 1]
    if bot_results:
        print(json.dumps(bot_results[0].to_dict(), indent=2))