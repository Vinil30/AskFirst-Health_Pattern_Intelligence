from collections import defaultdict

import math
import numpy as np
import pandas as pd

from utils.data_loader import build_user_timeline


FEATURE_COLUMNS = [
    "support_forward",
    "support_reverse",
    "precedence_ratio",
    "avg_lag_days",
    "std_lag_days",
    "effect_hit_rate",
    "cause_frequency",
    "effect_frequency",
    "lift_proxy",
    "first_cause_day",
    "first_effect_day",
]


def _clean_tags(tags):
    cleaned = []
    for t in tags:
        tag = str(t).strip().lower()
        if not tag:
            continue
        if "pattern" in tag:
            continue
        cleaned.append(tag)
    return cleaned


def _safe_std(values):
    if len(values) < 2:
        return 0.0
    return float(np.std(values))


def build_relation_candidates(user, max_days=84):
    timeline = build_user_timeline(user)
    all_tags = sorted({t for s in timeline for t in _clean_tags(s["tags"])})
    if len(all_tags) < 2:
        return pd.DataFrame()

    tag_sessions = defaultdict(list)
    for s in timeline:
        for t in _clean_tags(s["tags"]):
            tag_sessions[t].append(s)

    rows = []
    total_sessions = len(timeline)
    anchor = timeline[0]["timestamp"]
    for cause in all_tags:
        for effect in all_tags:
            if cause == effect:
                continue

            forward_lags = []
            reverse_lags = []
            evidence = []
            effect_sessions_hit = set()

            for effect_session in tag_sessions[effect]:
                effect_ts = effect_session["timestamp"]
                valid_causes = [
                    c
                    for c in tag_sessions[cause]
                    if 0 <= (effect_ts - c["timestamp"]).days <= max_days
                ]
                valid_reverse = [
                    c
                    for c in tag_sessions[cause]
                    if 1 <= (c["timestamp"] - effect_ts).days <= max_days
                ]

                if valid_causes:
                    nearest = sorted(
                        valid_causes,
                        key=lambda c: (effect_ts - c["timestamp"]).days,
                    )[0]
                    lag = (effect_ts - nearest["timestamp"]).days
                    forward_lags.append(lag)
                    effect_sessions_hit.add(effect_session["session_id"])
                    evidence.append(
                        {
                            "cause_session_id": nearest["session_id"],
                            "cause_week": nearest["week"],
                            "effect_session_id": effect_session["session_id"],
                            "effect_week": effect_session["week"],
                            "lag_days": lag,
                        }
                    )

                if valid_reverse:
                    nearest_r = sorted(
                        valid_reverse,
                        key=lambda c: (c["timestamp"] - effect_ts).days,
                    )[0]
                    reverse_lags.append((nearest_r["timestamp"] - effect_ts).days)

            support_forward = len(forward_lags)
            support_reverse = len(reverse_lags)
            precedence_ratio = (support_forward + 1.0) / (support_forward + support_reverse + 2.0)
            avg_lag = float(np.mean(forward_lags)) if forward_lags else 0.0
            std_lag = _safe_std(forward_lags)
            cause_freq = len(tag_sessions[cause])
            effect_freq = len(tag_sessions[effect])
            first_cause_day = min((s["timestamp"] - anchor).days for s in tag_sessions[cause])
            first_effect_day = min((s["timestamp"] - anchor).days for s in tag_sessions[effect])
            effect_hit_rate = len(effect_sessions_hit) / max(effect_freq, 1)
            expected_effect_rate = effect_freq / max(total_sessions, 1)
            observed_effect_rate = support_forward / max(cause_freq, 1)
            lift_proxy = observed_effect_rate / max(expected_effect_rate, 1e-6)

            rows.append(
                {
                    "user_id": user["user_id"],
                    "cause_tag": cause,
                    "effect_tag": effect,
                    "support_forward": float(support_forward),
                    "support_reverse": float(support_reverse),
                    "precedence_ratio": float(precedence_ratio),
                    "avg_lag_days": float(avg_lag),
                    "std_lag_days": float(std_lag),
                    "effect_hit_rate": float(effect_hit_rate),
                    "cause_frequency": float(cause_freq),
                    "effect_frequency": float(effect_freq),
                    "lift_proxy": float(lift_proxy),
                    "first_cause_day": float(first_cause_day),
                    "first_effect_day": float(first_effect_day),
                    "evidence": evidence[:4],
                }
            )

    return pd.DataFrame(rows)


def make_weak_labels(candidates):
    if candidates.empty:
        return candidates

    labeled = candidates.copy()
    pos = (
        (labeled["support_forward"] >= 2)
        & (labeled["precedence_ratio"] >= 0.58)
        & (labeled["lift_proxy"] >= 1.05)
    )
    neg = (labeled["support_forward"] == 0) | (labeled["precedence_ratio"] <= 0.45)
    labeled["target"] = np.where(pos, 1, np.where(neg, 0, np.nan))
    return labeled.dropna(subset=["target"]).copy()


def fallback_confidence_score(row):
    score = 0.0
    score += min(row["support_forward"] / 4.0, 1.0) * 0.35
    score += row["precedence_ratio"] * 0.25
    score += min(row["effect_hit_rate"], 1.0) * 0.15
    score += min(row["lift_proxy"] / 3.0, 1.0) * 0.2
    score += min(row["cause_frequency"] / 3.0, 1.0) * 0.05
    score *= math.exp(-max(row["avg_lag_days"] - 14.0, 0.0) / 40.0)
    return max(0.01, min(score, 0.99))


def confidence_label(p):
    if p >= 0.8:
        return "high"
    if p >= 0.6:
        return "medium"
    return "low"


def one_line_justification(row, confidence):
    return (
        f"{row['support_forward']:.0f} forward temporal matches with "
        f"precedence ratio {row['precedence_ratio']:.2f}, average lag {row['avg_lag_days']:.1f} days, "
        f"and lift proxy {row['lift_proxy']:.2f} support this relation."
    )


def reasoning_line(row):
    if not row.get("evidence"):
        return "No temporal evidence."
    e = row["evidence"][0]
    return (
        f"{row['cause_tag']} in {e['cause_session_id']} (Week {e['cause_week']}) precedes "
        f"{row['effect_tag']} in {e['effect_session_id']} (Week {e['effect_week']}) by "
        f"{e['lag_days']} days."
    )
