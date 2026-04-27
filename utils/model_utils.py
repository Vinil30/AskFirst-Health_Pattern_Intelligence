import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_dataset
from utils.feature_engineering import (
    FEATURE_COLUMNS,
    build_relation_candidates,
    confidence_label,
    fallback_confidence_score,
    make_weak_labels,
    one_line_justification,
    reasoning_line,
)


MODEL_PATH = Path("utils/relation_model.pkl")


def _train_pipeline(train_df):
    X = train_df[FEATURE_COLUMNS]
    y = train_df["target"].astype(int)
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )
    clf.fit(X, y)
    return clf


def train_and_save_model(dataset_path, model_path=MODEL_PATH):
    dataset = load_dataset(dataset_path)
    all_candidates = []
    for user in dataset["users"]:
        cand = build_relation_candidates(user)
        if not cand.empty:
            all_candidates.append(cand)

    if not all_candidates:
        raise ValueError("No candidates generated from dataset.")

    combined = pd.concat(all_candidates, ignore_index=True)
    labeled = make_weak_labels(combined)
    use_fallback_only = labeled.empty or labeled["target"].nunique() < 2

    bundle = {
        "feature_columns": FEATURE_COLUMNS,
        "trained_at": datetime.utcnow().isoformat(),
        "sklearn_version": sklearn.__version__,
        "use_fallback_only": bool(use_fallback_only),
        "training_rows": int(len(labeled)),
        "candidate_rows": int(len(combined)),
    }

    if not use_fallback_only:
        model = _train_pipeline(labeled)
        bundle["model"] = model
    else:
        bundle["model"] = None

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    return {
        "model_path": str(model_path),
        "trained_at": bundle["trained_at"],
        "fallback_only": bundle["use_fallback_only"],
        "training_rows": bundle["training_rows"],
        "candidate_rows": bundle["candidate_rows"],
    }


def load_model_bundle(model_path=MODEL_PATH):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    trained_version = bundle.get("sklearn_version")
    current_version = sklearn.__version__
    if trained_version and trained_version != current_version:
        raise RuntimeError(
            f"Model was trained with scikit-learn {trained_version} but current environment is {current_version}."
        )
    return bundle


def _predict_confidence(candidates, model_bundle):
    if candidates.empty:
        return candidates

    scored = candidates.copy()
    rule_scores = scored.apply(fallback_confidence_score, axis=1)
    if model_bundle.get("use_fallback_only") or model_bundle.get("model") is None:
        scored["confidence"] = rule_scores
    else:
        probs = model_bundle["model"].predict_proba(scored[model_bundle["feature_columns"]])[:, 1]
        scored["confidence"] = (0.25 * probs) + (0.75 * rule_scores)
    scored["confidence_label"] = scored["confidence"].apply(confidence_label)
    return scored


def score_user_patterns(user, model_bundle, min_confidence=0.55):
    candidates = build_relation_candidates(user)
    if candidates.empty:
        return {"patterns": [], "reasoning_trace": [], "suggestions": []}

    scored = _predict_confidence(candidates, model_bundle)
    filtered = scored[
        (scored["confidence"] >= min_confidence) & (scored["support_forward"] >= 1)
    ].copy()
    filtered = filtered[
        (filtered["precedence_ratio"] >= 0.5)
        & (filtered["effect_frequency"] >= 2)
        & (filtered["avg_lag_days"] <= 70)
        & (filtered["first_cause_day"] <= (filtered["first_effect_day"] + 7))
    ]
    filtered = filtered.sort_values(["confidence", "support_forward"], ascending=False)

    patterns = []
    reasoning_trace = []
    seen_pairs = set()
    for _, row in filtered.iterrows():
        pair_key = tuple(sorted([row["cause_tag"], row["effect_tag"]]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        if len(patterns) >= 8:
            break
        confidence = float(np.clip(row["confidence"], 0.01, 0.99))
        patterns.append(
            {
                "relation": f"{row['cause_tag']} -> {row['effect_tag']}",
                "confidence": round(confidence, 4),
                "confidence_label": row["confidence_label"],
                "justification": one_line_justification(row, confidence),
            }
        )
        reasoning_trace.append(
            {
                "relation": f"{row['cause_tag']} -> {row['effect_tag']}",
                "trace": reasoning_line(row),
                "evidence": row["evidence"],
            }
        )

    suggestions = build_user_improvement_suggestions(patterns)
    return {
        "patterns": patterns,
        "reasoning_trace": reasoning_trace,
        "suggestions": suggestions,
    }


def build_user_improvement_suggestions(patterns):
    if not patterns:
        return [
            "Insufficient repeated temporal evidence; collect more sessions before high-confidence conclusions.",
            "Capture objective markers (sleep hours, hydration amount, meal timing) to reduce ambiguity.",
        ]

    high = [p for p in patterns if p["confidence_label"] == "high"]
    medium = [p for p in patterns if p["confidence_label"] == "medium"]
    return [
        "Track daily lifestyle variables (sleep time, hydration, meal timing, stress level) in structured form to strengthen causal confidence.",
        "Run a contradiction check by searching sessions where suspected causes appear without effects and vice versa.",
        f"Current pattern mix: {len(high)} high-confidence and {len(medium)} medium-confidence relations. Reassess low-confidence links with more data.",
        "With more time, add NER and medical ontology normalization so inconsistent wording maps to the same clinical concepts.",
    ]
