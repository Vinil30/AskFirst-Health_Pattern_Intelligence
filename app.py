import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.data_loader import load_dataset
from utils.groq_structurer import stream_structured_output
from utils.model_utils import (
    MODEL_PATH,
    load_model_bundle,
    score_user_patterns,
    train_and_save_model,
)

DATASET_PATH = Path("askfirst_synthetic_dataset.json")
load_dotenv()

# ── Injected CSS ──────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;0,700;1,400&family=Source+Sans+3:wght@300;400;500;600&family=Source+Code+Pro:wght@400;500&display=swap');

/* ── Root palette — clinical warmth ─────────────────────── */
:root {
    --bg:        #f7f5f0;
    --surface:   #ffffff;
    --surface2:  #f0ede6;
    --border:    #ddd8ce;
    --border2:   #c8c2b6;
    --accent:    #3a7d5e;       /* sage / medical green */
    --accent2:   #2a6097;       /* trustworthy slate blue */
    --warm:      #b35c2e;       /* amber for warnings */
    --muted:     #8a8278;
    --text:      #2c2a26;
    --subtext:   #6b6660;
    --card-bg:   #fdfcf9;
    --green-lt:  #eaf4ef;
    --blue-lt:   #e8f0f8;
    --amber-lt:  #fdf3ec;
}

/* ── Global reset ────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ───────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.2rem 2.8rem 4rem !important;
    max-width: 1240px !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stButton label { color: var(--subtext) !important; }

/* ── Title ───────────────────────────────────────────────── */
h1 {
    font-family: 'Lora', Georgia, serif !important;
    font-weight: 700 !important;
    font-size: 2.15rem !important;
    color: #ffffff !important;
    letter-spacing: -0.3px !important;
    line-height: 1.25 !important;
    margin-bottom: 0 !important;
}
.caption-bar {
    color: var(--subtext);
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.8rem;
    font-weight: 400;
    letter-spacing: 0.04em;
    margin-top: 0.35rem;
    margin-bottom: 2rem;
    padding-left: 0.9rem;
    border-left: 3px solid var(--accent);
}

/* ── Section headers ─────────────────────────────────────── */
h2 {
    font-family: 'Lora', Georgia, serif !important;
    font-weight: 600 !important;
    font-size: 1.3rem !important;
    color: var(--text) !important;
    margin-top: 2rem !important;
    margin-bottom: 0.6rem !important;
}
h3 {
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    margin-top: 1.6rem !important;
    margin-bottom: 0.5rem !important;
}

/* ── Info card ───────────────────────────────────────────── */
.info-card {
    background: var(--blue-lt);
    border: 1px solid #c0d4e8;
    border-left: 4px solid var(--accent2);
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.84rem;
    line-height: 1.65;
    color: var(--subtext);
    font-family: 'Source Sans 3', sans-serif;
}
.info-card strong { color: var(--accent2); font-weight: 600; }

/* ── Step badge row ──────────────────────────────────────── */
.steps-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin: 0.8rem 0 1.8rem;
}
.step-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.4rem 0.9rem;
    font-size: 0.76rem;
    color: var(--subtext);
    font-family: 'Source Sans 3', sans-serif;
}
.step-badge .num {
    background-color: var(--accent);
    color: #fff;
    font-weight: 600;
    border-radius: 50%;
    width: 19px; height: 19px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.62rem;
    flex-shrink: 0;
}

/* ── Stat tiles ──────────────────────────────────────────── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
    gap: 0.7rem;
    margin: 1rem 0;
}
.stat-tile {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-tile .val {
    font-family: 'Lora', Georgia, serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.stat-tile .label {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-top: 0.35rem;
    font-family: 'Source Sans 3', sans-serif;
}

/* ── User result card ────────────────────────────────────── */
.user-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 6px;
    padding: 0.9rem 1.3rem;
    margin: 1.6rem 0 0.6rem;
}
.user-avatar {
    width: 40px; height: 40px;
    border-radius: 50%;
    background-color: var(--green-lt);
    border: 2px solid var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Lora', serif;
    font-weight: 700;
    color: var(--accent);
    font-size: 0.95rem;
    flex-shrink: 0;
}
.user-name {
    font-family: 'Lora', Georgia, serif;
    font-weight: 600;
    font-size: 1rem;
    color: var(--text);
}
.user-id {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.1rem;
    font-family: 'Source Code Pro', monospace;
}

/* ── Section labels ──────────────────────────────────────── */
.result-section-label {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.11em;
    color: var(--accent);
    background: var(--green-lt);
    border: 1px solid #b8ddc9;
    border-radius: 4px;
    padding: 0.22rem 0.65rem;
    margin: 0.9rem 0 0.35rem;
    font-family: 'Source Sans 3', sans-serif;
}
.result-section-label.blue {
    color: var(--accent2);
    background: var(--blue-lt);
    border-color: #b0c8e0;
}
.result-section-label.amber {
    color: var(--warm);
    background: var(--amber-lt);
    border-color: #e4c4ae;
}

/* ── Code / JSON blocks ──────────────────────────────────── */
.stCodeBlock > div, pre, code {
    background: #f4f2ed !important;
    border: 1px solid var(--border) !important;
    border-radius: 5px !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.77rem !important;
    color: #2e5c3e !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background-color: var(--accent) !important;
    color: #ffffff !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.5rem 1.4rem !important;
    transition: background-color 0.18s ease !important;
}
.stButton > button:hover {
    background-color: #2e6349 !important;
}

/* ── Download button ─────────────────────────────────────── */
.stDownloadButton > button {
    background: var(--surface) !important;
    color: var(--accent2) !important;
    border: 1px solid var(--accent2) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 5px !important;
}

/* ── Select box ──────────────────────────────────────────── */
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 5px !important;
    color: var(--text) !important;
}

/* ── Spinner / status ────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--accent) !important; }
.stAlert { border-radius: 6px !important; border-left-width: 4px !important; font-family: 'Source Sans 3', sans-serif !important; }

/* ── Expander ────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: var(--subtext) !important;
    background: var(--surface2) !important;
    border-radius: 5px !important;
}

/* ── Divider ─────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.6rem 0 !important;
}

/* ── Streaming output box ────────────────────────────────── */
.stream-box {
    background: #fdfcf9;
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent2);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    min-height: 80px;
    font-family: 'Source Code Pro', monospace;
    font-size: 0.76rem;
    color: #2a5080;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 420px;
    overflow-y: auto;
    line-height: 1.55;
}
</style>
"""

STEPS = [
    "Sort user timeline by timestamp",
    "Build tag-pair candidate relations",
    "Extract support / lag / lift features",
    "Score via calibrated logistic model",
    "Groq restructures trace into JSON",
]


def _render_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _stream_to_placeholder(placeholder, chunks):
    streamed = ""
    for piece in chunks:
        streamed += piece
        placeholder.markdown(
            f'<div class="stream-box">{streamed}</div>', unsafe_allow_html=True
        )
    return streamed


def _get_selected_users(dataset, selected_user):
    users = dataset["users"]
    if selected_user == "ALL":
        return users
    return [u for u in users if u["user_id"] == selected_user]


def _user_initials(name: str) -> str:
    parts = name.strip().split()
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper()


def _render_steps():
    badges = "".join(
        f'<div class="step-badge"><span class="num">{i+1}</span>{s}</div>'
        for i, s in enumerate(STEPS)
    )
    st.markdown(f'<div class="steps-row">{badges}</div>', unsafe_allow_html=True)


def _render_user_result(user, result):
    initials = _user_initials(user["name"])
    st.markdown(
        f"""
        <div class="user-header">
            <div class="user-avatar">{initials}</div>
            <div>
                <div class="user-name">{user["name"]}</div>
                <div class="user-id">{user["user_id"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="result-section-label">⚡ Reasoning Trace</div>',
            unsafe_allow_html=True,
        )
        with st.expander("View trace", expanded=False):
            st.json(result["reasoning_trace"])

    with col2:
        st.markdown(
            '<div class="result-section-label blue">🔬 Detected Patterns</div>',
            unsafe_allow_html=True,
        )
        with st.expander("View patterns", expanded=False):
            st.json(result["patterns"])

    st.markdown(
        '<div class="result-section-label amber">💡 Improvement Suggestions</div>',
        unsafe_allow_html=True,
    )
    with st.expander("View suggestions", expanded=True):
        st.json(result["suggestions"])

    st.markdown("<hr/>", unsafe_allow_html=True)


def main():
    _render_css()

    st.set_page_config(
        page_title="Ask First · Pattern Detector",
        page_icon="🩺",
        layout="wide",
    )

    st.markdown("# Ask First · Health Pattern Detector")
    st.markdown(
        '<div class="caption-bar">Temporal cross-conversation pattern detection &nbsp;·&nbsp; '
        'Confidence scoring &nbsp;·&nbsp; Groq JSON structuring</div>',
        unsafe_allow_html=True,
    )

    if not DATASET_PATH.exists():
        st.error(f"❌ Dataset not found: `{DATASET_PATH}`")
        return

    dataset = load_dataset(DATASET_PATH)
    users = dataset["users"]

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<h2 style="font-family:Lora,Georgia,serif;font-size:1.05rem;font-weight:600;'
            'color:#2c2a26;margin-bottom:1rem;letter-spacing:-0.2px;">Configuration</h2>',
            unsafe_allow_html=True,
        )
        selected_user = st.selectbox(
            "Select user",
            options=["ALL"] + [u["user_id"] for u in users],
            index=0,
        )
        st.markdown(
            f'<div style="font-size:0.72rem;color:#8a8278;margin-top:-0.4rem;'
            f'margin-bottom:1rem;">{len(users)} users in dataset</div>',
            unsafe_allow_html=True,
        )
        retrain = st.button("🔄 Retrain model")
        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.7rem;color:#8a8278;line-height:1.6;">'
            'Model artifact versioned at<br>'
            f'<code style="color:#3a7d5e;font-family:\'Source Code Pro\',monospace">{MODEL_PATH}</code></div>',
            unsafe_allow_html=True,
        )

    # ── Groq keys ─────────────────────────────────────────────────────────────
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not groq_api_key:
        st.markdown(
            '<div class="info-card">ℹ️ <strong>GROQ_API_KEY</strong> not found in '
            '<code>.env</code>. App will fall back to local JSON structuring.</div>',
            unsafe_allow_html=True,
        )

    # ── Pipeline overview ──────────────────────────────────────────────────────
    st.markdown("### Pipeline overview")
    _render_steps()

    # ── Model training ─────────────────────────────────────────────────────────
    if retrain or not MODEL_PATH.exists():
        with st.spinner("Training relation model…"):
            stats = train_and_save_model(DATASET_PATH, MODEL_PATH)
        st.success(f"✅ Model saved → `{MODEL_PATH}`")

        keys = list(stats.keys())
        tile_html = '<div class="stat-grid">' + "".join(
            f'<div class="stat-tile"><div class="val">{stats[k]}</div>'
            f'<div class="label">{k}</div></div>'
            for k in keys
        ) + "</div>"
        st.markdown(tile_html, unsafe_allow_html=True)

    if not MODEL_PATH.exists():
        st.warning("⚠ Model artifact missing — click **Retrain model** in the sidebar.")
        return

    # ── Load model ─────────────────────────────────────────────────────────────
    try:
        model_bundle = load_model_bundle(MODEL_PATH)
    except RuntimeError as e:
        st.warning(f"⚠ {e}  Re-training…")
        with st.spinner("Re-training to fix version mismatch…"):
            stats = train_and_save_model(DATASET_PATH, MODEL_PATH)
        st.info(f"Model retrained: {stats}")
        model_bundle = load_model_bundle(MODEL_PATH)

    # ── Run button ─────────────────────────────────────────────────────────────
    st.markdown("---")
    run = st.button("▶ Run temporal reasoning")

    if run:
        selected_users = _get_selected_users(dataset, selected_user)

        st.markdown(
            f'<div class="info-card">Running on <strong>{len(selected_users)}</strong> '
            f'user(s) — analysing temporal tag relations…</div>',
            unsafe_allow_html=True,
        )

        all_raw = []
        all_suggestions = []

        for user in selected_users:
            with st.spinner(f"Scoring patterns for {user['name']}…"):
                result = score_user_patterns(user, model_bundle)
            _render_user_result(user, result)

            all_raw.append(
                {
                    "user_id": user["user_id"],
                    "name": user["name"],
                    "patterns": result["patterns"],
                    "reasoning_trace": result["reasoning_trace"],
                }
            )
            all_suggestions.append(
                {"user_id": user["user_id"], "suggestions": result["suggestions"]}
            )

        payload = {
            "task": "Structure temporal health relations into strict JSON with confidence and justification.",
            "users": all_raw,
            "suggestions": all_suggestions,
            "requirements": {
                "strict_json": True,
                "include_confidence_reason": True,
                "include_loopholes_and_improvements": True,
            },
        }

        st.markdown("---")
        st.markdown("### 🤖 Groq structured output")
        st.markdown(
            '<div class="info-card">Streaming structured JSON from Groq — '
            f'model: <strong>{groq_model}</strong></div>',
            unsafe_allow_html=True,
        )

        placeholder = st.empty()
        final_json = _stream_to_placeholder(
            placeholder,
            stream_structured_output(
                payload=payload,
                api_key=groq_api_key,
                model=groq_model,
            ),
        )

        st.markdown("<br/>", unsafe_allow_html=True)
        st.download_button(
            label="⬇ Download structured JSON",
            data=final_json,
            file_name="structured_output.json",
            mime="application/json",
        )


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is None and len(sys.argv) == 1:
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", __file__]
        raise SystemExit(stcli.main())
    main()