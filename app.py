import os
import re
import csv
import random
import time
from dataclasses import dataclass
from datetime import datetime
from math import erfc, sqrt
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI Reliability Experiment", page_icon="🧪", layout="wide")
st.title("AI Reliability Experiment")
st.caption("Testing whether structured reasoning improves AI fact-checking reliability")

# -----------------------------
# Locked model settings
# -----------------------------
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_TOKENS = 800

SOFT_CALL_LIMIT = 200  # safety guard
COST_EST_PER_CALL = 0.002  # rough estimate

API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY) if API_KEY else None

# -----------------------------
# Dataset (20 claims)
# -----------------------------
CLAIMS = [
    {"text": "If Earth stopped rotating but continued orbiting the Sun, we would still have normal day and night cycles.", "answer": "False"},
    {"text": "If you double the mass of an object in free fall in a vacuum, it will hit the ground at the same time as a lighter object.", "answer": "True"},
    {"text": "Increasing the temperature of a gas in a sealed container increases the pressure.", "answer": "True"},
    {"text": "If two objects have the same volume but different masses, they must have the same density.", "answer": "False"},
    {"text": "An object moving at constant velocity has no net force acting on it.", "answer": "True"},
    {"text": "If a plant is placed in the dark, it immediately stops cellular respiration.", "answer": "False"},
    {"text": "In space, a rocket must continuously fire its engines to keep moving forward.", "answer": "False"},
    {"text": "If two chemicals react and temperature decreases, the reaction cannot be exothermic.", "answer": "True"},
    {"text": "If you increase the speed of a moving object, its kinetic energy increases linearly.", "answer": "False"},
    {"text": "A metal object and a wooden object of the same mass will experience the same gravitational force near Earth’s surface.", "answer": "True"},
    {"text": "If Earth had no atmosphere, the sky would appear blue during the day.", "answer": "False"},
    {"text": "When ice melts, the total mass of the water remains the same.", "answer": "True"},
    {"text": "If a chemical reaction absorbs heat from its surroundings, it is exothermic.", "answer": "False"},
    {"text": "An object in orbit around Earth is constantly falling toward Earth.", "answer": "True"},
    {"text": "If two planets have the same mass but different radii, the one with the smaller radius has stronger surface gravity.", "answer": "True"},
    {"text": "If light travels through water instead of air, its wavelength stays exactly the same.", "answer": "False"},
    {"text": "A higher pH value means a solution is more acidic.", "answer": "False"},
    {"text": "Viruses are considered living organisms.", "answer": "Uncertain"},
    {"text": "Pluto is a planet.", "answer": "Uncertain"},
    {"text": "Glass is a solid.", "answer": "Uncertain"},
]

# -----------------------------
# Prompts
# -----------------------------
SYSTEM_PROMPT = (
    "You are a careful science fact-checker for a 6th grade project. "
    "Classify the claim as True, False, or Uncertain. Use clear, simple language."
)

QUICK_PROMPT = """
Respond EXACTLY in this format:

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>%
EXPLANATION: <2-3 sentences>

Claim: "{claim}"
"""

REASON_PROMPT = """
You MUST reason step-by-step before deciding.

Respond EXACTLY in this format:

REASONING:
- Step 1: Break the claim into parts.
- Step 2: Identify relevant science concepts.
- Step 3: Compare the claim to known scientific principles.

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>%
EXPLANATION: <2-3 sentences>

Claim: "{claim}"
"""

# -----------------------------
# Session state
# -----------------------------
def init_state():
    if "page" not in st.session_state:
        st.session_state.page = "Experiment"

    if "idx" not in st.session_state:
        st.session_state.idx = 0

    if "results" not in st.session_state:
        # results[mode][trial_key] = AIResult
        st.session_state.results = {"quick": {}, "reason": {}}

    if "api_calls" not in st.session_state:
        st.session_state.api_calls = 0

    if "quota_warning" not in st.session_state:
        st.session_state.quota_warning = False

    # Persist last judge batch results for display
    if "last_batch" not in st.session_state:
        st.session_state.last_batch = None  # dict with indices and outputs

    # Persist Ask-your-own outputs
    if "ask_last" not in st.session_state:
        st.session_state.ask_last = None  # dict with claim + quick + reason

init_state()

# Reproducibility seed
with st.expander("⚙️ Controls (reproducibility + session usage)", expanded=False):
    seed = st.number_input("Random Seed (reproducibility)", value=42, step=1)
    random.seed(int(seed))
    est_cost = st.session_state.api_calls * COST_EST_PER_CALL
    st.write(f"API calls this session: **{st.session_state.api_calls}** (soft limit: {SOFT_CALL_LIMIT})")
    st.write(f"Estimated cost (rough): **${est_cost:.4f}**")
    st.write(f"Locked settings: model={MODEL}, temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}")
    if st.session_state.quota_warning:
        st.warning("A quota/rate-limit warning happened this session.")

# -----------------------------
# Helpers
# -----------------------------
@dataclass
class AIResult:
    verdict: str
    confidence: int
    explanation: str
    raw_text: str
    seconds: float

def clamp_conf(x: int) -> int:
    return max(0, min(100, int(x)))

def parse_fields(text: str) -> Tuple[str, int, str]:
    verdict = "Uncertain"
    confidence = 50
    explanation = ""

    v = re.search(r"VERDICT:\s*(True|False|Uncertain)", text, re.I)
    if v:
        verdict = v.group(1).capitalize()

    c = re.search(r"CONFIDENCE:\s*(\d+)", text, re.I)
    if c:
        confidence = clamp_conf(int(c.group(1)))

    e = re.search(r"EXPLANATION:\s*(.+)", text, re.I | re.S)
    if e:
        explanation = e.group(1).strip()

    return verdict, confidence, explanation

def badge(verdict: str) -> str:
    return {"True": "🟢 TRUE", "False": "🔴 FALSE"}.get(verdict, "🟡 UNCERTAIN")

def is_correct(verdict: str, truth: str) -> bool:
    return verdict == truth

def call_ai(claim: str, mode: str) -> AIResult:
    if client is None:
        return AIResult("Uncertain", 0, "OPENAI_API_KEY is missing.", "", 0.0)

    if st.session_state.api_calls >= SOFT_CALL_LIMIT:
        st.session_state.quota_warning = True
        return AIResult("Uncertain", 0, "Soft API call limit reached for this session.", "", 0.0)

    prompt = QUICK_PROMPT if mode == "quick" else REASON_PROMPT
    formatted = prompt.format(claim=claim)

    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": formatted},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        t1 = time.time()
        st.session_state.api_calls += 1

        text = resp.choices[0].message.content or ""
        verdict, conf, expl = parse_fields(text)
        return AIResult(verdict, conf, expl, text, round(t1 - t0, 2))

    except Exception as e:
        msg = str(e)
        if "429" in msg or "insufficient_quota" in msg:
            st.session_state.quota_warning = True
        return AIResult("Uncertain", 0, "API error/rate limit. Check your plan/billing.", msg, 0.0)

def add_bar_labels(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            f"{h:.1f}",
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

def summarize_mode(mode: str) -> Dict[str, float]:
    """Returns dict with n, acc, avg_conf, cal_gap, unc_rate"""
    data = st.session_state.results[mode]
    if not data:
        return {"n": 0, "acc": 0.0, "avg_conf": 0.0, "cal_gap": 0.0, "unc_rate": 0.0}

    # We only score one row per claim index for now (latest stored per claim)
    n = len(data)
    correct = 0
    conf_sum = 0
    unc = 0

    for claim_idx, res in data.items():
        truth = CLAIMS[claim_idx]["answer"]
        if is_correct(res.verdict, truth):
            correct += 1
        conf_sum += res.confidence
        if res.verdict == "Uncertain":
            unc += 1

    acc = (correct / n) * 100.0
    avg_conf = conf_sum / n
    cal_gap = abs(avg_conf - acc)
    unc_rate = (unc / n) * 100.0
    return {"n": n, "acc": acc, "avg_conf": avg_conf, "cal_gap": cal_gap, "unc_rate": unc_rate}

def mcnemar_approx_p() -> Optional[Dict[str, float]]:
    q = st.session_state.results["quick"]
    r = st.session_state.results["reason"]
    shared = sorted(set(q.keys()) & set(r.keys()))
    if len(shared) < 2:
        return None

    b = 0
    c = 0
    for i in shared:
        truth = CLAIMS[i]["answer"]
        q_ok = is_correct(q[i].verdict, truth)
        r_ok = is_correct(r[i].verdict, truth)
        if q_ok and (not r_ok):
            b += 1
        elif (not q_ok) and r_ok:
            c += 1

    if (b + c) == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}

    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)  # continuity correction
    p = erfc(sqrt(chi2 / 2.0))  # df=1 approx
    return {"b": float(b), "c": float(c), "chi2": float(chi2), "p": float(p)}

# -----------------------------
# Navigation
# -----------------------------
st.session_state.page = st.radio(
    "Choose a section:",
    ["Experiment", "Ask Your Own Question", "Results"],
    horizontal=True,
    index=["Experiment", "Ask Your Own Question", "Results"].index(st.session_state.page),
)

st.markdown("---")

# ============================================================
# PAGE: EXPERIMENT (Manual 20 + Judge Mode 5 random)
# ============================================================
if st.session_state.page == "Experiment":
    st.subheader("🔬 Experiment (Manual 20-Claim Test)")

    st.info(
        "Goal: Compare **Quick Mode** vs **Reasoning Mode** on the same claims.\n\n"
        "**Independent variable:** response mode\n"
        "**Dependent variables:** accuracy, confidence, calibration gap, uncertainty rate"
    )

    # --- Judge demo controls ---
    st.markdown("### Judge Demo Controls")
    st.caption("These run multiple claims automatically and DISPLAY the questions + results on screen.")

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        if st.button("🎲 Run 5 Random Claims (Judge Mode)", use_container_width=True):
            indices = random.sample(range(len(CLAIMS)), 5)
            batch_rows = []

            for i in indices:
                claim_text = CLAIMS[i]["text"]
                truth = CLAIMS[i]["answer"]

                quick = call_ai(claim_text, "quick")
                reason = call_ai(claim_text, "reason")

                # Save to main dataset
                st.session_state.results["quick"][i] = quick
                st.session_state.results["reason"][i] = reason

                batch_rows.append({
                    "idx": i,
                    "claim": claim_text,
                    "truth": truth,
                    "quick": quick,
                    "reason": reason
                })

            st.session_state.last_batch = {"indices": indices, "rows": batch_rows, "timestamp": datetime.now().isoformat()}
            st.success("Completed 5 random claims. Scroll down to see the questions and results.")
            st.rerun()

    with colB:
        if st.button("🧪 Start / Restart Manual 20-Claim Experiment", use_container_width=True):
            st.session_state.idx = 0
            st.session_state.last_batch = None
            st.success("Manual experiment ready. Scroll down to Claim 1.")
            st.rerun()

    with colC:
        if st.button("🧹 Clear ALL Saved Results", use_container_width=True):
            st.session_state.results = {"quick": {}, "reason": {}}
            st.session_state.idx = 0
            st.session_state.last_batch = None
            st.session_state.ask_last = None
            st.session_state.api_calls = 0
            st.session_state.quota_warning = False
            st.success("Cleared all results.")
            st.rerun()

    # --- Display judge batch results (so it's NOT "running in the background") ---
    if st.session_state.last_batch:
        st.markdown("---")
        st.subheader("✅ Judge Mode Output (5 Random Claims)")
        st.caption(f"Generated at: {st.session_state.last_batch['timestamp']}")

        for n, row in enumerate(st.session_state.last_batch["rows"], start=1):
            i = row["idx"]
            st.markdown(f"### {n}) Claim #{i+1}")
            st.write(row["claim"])
            st.write(f"Ground truth: **{row['truth']}**")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Quick Mode**")
                st.write(badge(row["quick"].verdict))
                st.write(f"Confidence: **{row['quick'].confidence}%** • Time: **{row['quick'].seconds}s**")
                st.write(f"Correct: **{is_correct(row['quick'].verdict, row['truth'])}**")
                st.write(row["quick"].explanation)

            with c2:
                st.markdown("**Reasoning Mode**")
                st.write(badge(row["reason"].verdict))
                st.write(f"Confidence: **{row['reason'].confidence}%** • Time: **{row['reason'].seconds}s**")
                st.write(f"Correct: **{is_correct(row['reason'].verdict, row['truth'])}**")
                # show reasoning bullets if present
                if "REASONING:" in row["reason"].raw_text:
                    reasoning_part = row["reason"].raw_text.split("VERDICT:")[0].strip()
                    st.text(reasoning_part)
                st.write(row["reason"].explanation)

            st.markdown("---")

    # --- Manual experiment section ---
    st.markdown("---")
    st.subheader("🧭 Manual Claim Runner (20 claims, step-by-step)")

    idx = st.session_state.idx
    claim_text = CLAIMS[idx]["text"]
    truth = CLAIMS[idx]["answer"]

    st.progress((idx + 1) / len(CLAIMS))
    st.markdown(f"### Claim {idx+1}/{len(CLAIMS)}")
    st.write(claim_text)
    st.write(f"Ground truth: **{truth}**")

    run_col1, run_col2 = st.columns(2)
    with run_col1:
        if st.button("⚡ Run Quick Mode for this claim", use_container_width=True):
            st.session_state.results["quick"][idx] = call_ai(claim_text, "quick")
            st.rerun()

    with run_col2:
        if st.button("🧠 Run Reasoning Mode for this claim", use_container_width=True):
            st.session_state.results["reason"][idx] = call_ai(claim_text, "reason")
            st.rerun()

    # Display current claim outputs
    st.markdown("#### Outputs for this claim")
    q_res = st.session_state.results["quick"].get(idx)
    r_res = st.session_state.results["reason"].get(idx)

    out1, out2 = st.columns(2)
    with out1:
        st.markdown("**Quick Mode**")
        if q_res:
            st.write(badge(q_res.verdict))
            st.write(f"Confidence: **{q_res.confidence}%** • Time: **{q_res.seconds}s**")
            st.write(f"Correct: **{is_correct(q_res.verdict, truth)}**")
            st.write(q_res.explanation)
        else:
            st.info("Run Quick Mode to see output.")

    with out2:
        st.markdown("**Reasoning Mode**")
        if r_res:
            st.write(badge(r_res.verdict))
            st.write(f"Confidence: **{r_res.confidence}%** • Time: **{r_res.seconds}s**")
            st.write(f"Correct: **{is_correct(r_res.verdict, truth)}**")
            if "REASONING:" in r_res.raw_text:
                reasoning_part = r_res.raw_text.split("VERDICT:")[0].strip()
                st.text(reasoning_part)
            st.write(r_res.explanation)
        else:
            st.info("Run Reasoning Mode to see output.")

    # Next claim: only after both run
    both_done = (q_res is not None) and (r_res is not None)
    st.caption("Next Claim unlocks only after BOTH modes are run for the current claim.")
    if st.button("Next Claim ➜", disabled=not both_done):
        if idx < len(CLAIMS) - 1:
            st.session_state.idx += 1
            st.rerun()
        else:
            st.success("Manual experiment complete! Go to Results.")

# ============================================================
# PAGE: ASK YOUR OWN QUESTION (persist results)
# ============================================================
elif st.session_state.page == "Ask Your Own Question":
    st.subheader("🧠 Ask Your Own Question (Side-by-side)")

    user_claim = st.text_input("Enter a science claim:", placeholder="Example: The Moon has its own light.")
    if st.button("Analyze (run both modes)", use_container_width=True):
        if not user_claim.strip():
            st.warning("Please enter a claim.")
        else:
            quick = call_ai(user_claim.strip(), "quick")
            reason = call_ai(user_claim.strip(), "reason")
            st.session_state.ask_last = {"claim": user_claim.strip(), "quick": quick, "reason": reason}
            st.rerun()

    # Persistent display
    if st.session_state.ask_last:
        st.markdown("---")
        st.markdown("### Latest comparison")
        st.write(f"**Claim:** {st.session_state.ask_last['claim']}")

        c1, c2 = st.columns(2)
        with c1:
            q = st.session_state.ask_last["quick"]
            st.markdown("#### Quick Mode")
            st.write(badge(q.verdict))
            st.write(f"Confidence: **{q.confidence}%** • Time: **{q.seconds}s**")
            st.write(q.explanation)

        with c2:
            r = st.session_state.ask_last["reason"]
            st.markdown("#### Reasoning Mode")
            st.write(badge(r.verdict))
            st.write(f"Confidence: **{r.confidence}%** • Time: **{r.seconds}s**")
            if "REASONING:" in r.raw_text:
                reasoning_part = r.raw_text.split("VERDICT:")[0].strip()
                st.text(reasoning_part)
            st.write(r.explanation)

# ============================================================
# PAGE: RESULTS (dashboard + export)
# ============================================================
else:
    st.subheader("📊 Results Dashboard")

    quick = st.session_state.results["quick"]
    reason = st.session_state.results["reason"]
    shared = sorted(set(quick.keys()) & set(reason.keys()))

    st.write(f"Claims completed in BOTH modes: **{len(shared)}/{len(CLAIMS)}**")
    if len(shared) == 0:
        st.warning("Run at least one claim in BOTH modes to generate results.")
        st.stop()

    q_sum = summarize_mode("quick")
    r_sum = summarize_mode("reason")

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Quick Accuracy", f"{q_sum['acc']:.1f}%")
    m2.metric("Reasoning Accuracy", f"{r_sum['acc']:.1f}%")
    m3.metric("Quick Calibration Gap", f"{q_sum['cal_gap']:.1f}")
    m4.metric("Reasoning Calibration Gap", f"{r_sum['cal_gap']:.1f}")

    st.divider()

    # Small bar chart helper
    def small_bar(title: str, labels: List[str], values: List[float], ylabel: str):
        fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=120)
        bars = ax.bar(labels, values)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        add_bar_labels(ax, bars)
        fig.tight_layout()
        st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        small_bar("Accuracy (%)", ["Quick", "Reasoning"], [q_sum["acc"], r_sum["acc"]], "Percent")
    with c2:
        small_bar("Average Confidence (%)", ["Quick", "Reasoning"], [q_sum["avg_conf"], r_sum["avg_conf"]], "Percent")

    c3, c4 = st.columns(2)
    with c3:
        small_bar("Calibration Gap (lower is better)", ["Quick", "Reasoning"], [q_sum["cal_gap"], r_sum["cal_gap"]], "Abs(Conf − Acc)")
    with c4:
        small_bar("Uncertainty Rate (%)", ["Quick", "Reasoning"], [q_sum["unc_rate"], r_sum["unc_rate"]], "Percent")

    # Confidence vs correctness scatter (calibration view)
    st.markdown("### Confidence vs Correctness (Calibration View)")
    q_x = []
    q_y = []
    r_x = []
    r_y = []
    for i in shared:
        truth = CLAIMS[i]["answer"]
        q_ok = 1 if is_correct(quick[i].verdict, truth) else 0
        r_ok = 1 if is_correct(reason[i].verdict, truth) else 0
        q_x.append(quick[i].confidence)
        q_y.append(q_ok)
        r_x.append(reason[i].confidence)
        r_y.append(r_ok)

    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=120)
    ax.scatter(q_x, q_y, label="Quick")
    ax.scatter(r_x, r_y, label="Reasoning")
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Correct (1=yes, 0=no)")
    ax.set_yticks([0, 1])
    ax.set_title("Higher confidence should align with correctness")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    # Statistical significance approx
    st.markdown("### Statistical significance approximation (paired)")
    sig = mcnemar_approx_p()
    if sig is None:
        st.write("Not enough paired results yet.")
    else:
        st.write(f"Disagreements: **b={int(sig['b'])}** (Quick correct, Reasoning wrong) • **c={int(sig['c'])}** (Quick wrong, Reasoning correct)")
        st.write(f"χ² (approx): **{sig['chi2']:.3f}** • p-value (approx): **{sig['p']:.3f}**")
        st.caption("McNemar-style paired approximation (best effort; small sample sizes are approximate).")

    # Narrative summary
    st.markdown("---")
    st.markdown("### Results summary (for judges)")
    better_acc = "Reasoning" if r_sum["acc"] > q_sum["acc"] else ("Quick" if q_sum["acc"] > r_sum["acc"] else "Tie")
    better_gap = "Reasoning" if r_sum["cal_gap"] < q_sum["cal_gap"] else ("Quick" if q_sum["cal_gap"] < r_sum["cal_gap"] else "Tie")

    st.write(
        f"Across **{len(shared)}** paired claims, **{better_acc} Mode** had higher accuracy "
        f"({q_sum['acc']:.1f}% Quick vs {r_sum['acc']:.1f}% Reasoning). "
        f"Calibration was better in **{better_gap} Mode** "
        f"(gap {q_sum['cal_gap']:.1f} Quick vs {r_sum['cal_gap']:.1f} Reasoning; lower is better). "
        f"Uncertainty rate was {q_sum['unc_rate']:.1f}% (Quick) vs {r_sum['unc_rate']:.1f}% (Reasoning)."
    )

    # CSV export (paired only)
    st.markdown("### Download data (CSV)")
    rows = []
    now = datetime.now().isoformat()
    for i in shared:
        truth = CLAIMS[i]["answer"]
        q_res = quick[i]
        r_res = reason[i]
        rows.append([
            now,
            i + 1,
            CLAIMS[i]["text"],
            truth,
            q_res.verdict,
            q_res.confidence,
            is_correct(q_res.verdict, truth),
            r_res.verdict,
            r_res.confidence,
            is_correct(r_res.verdict, truth),
        ])

    csv_path = "/mnt/data/experiment_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp",
            "claim_number",
            "claim_text",
            "ground_truth",
            "quick_verdict",
            "quick_confidence",
            "quick_correct",
            "reason_verdict",
            "reason_confidence",
            "reason_correct",
        ])
        w.writerows(rows)

    with open(csv_path, "rb") as f:
        st.download_button("Download Experiment Data (CSV)", f, file_name="experiment_results.csv", mime="text/csv")

    # Reset button
    st.markdown("---")
    if st.button("Reset Experiment (clear all saved results)"):
        st.session_state.results = {"quick": {}, "reason": {}}
        st.session_state.idx = 0
        st.session_state.last_batch = None
        st.session_state.ask_last = None
        st.session_state.api_calls = 0
        st.session_state.quota_warning = False
        st.success("Reset complete.")
        st.rerun()