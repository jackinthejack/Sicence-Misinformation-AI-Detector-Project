import os
import re
import random
import statistics
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="AI Reliability Experiment", page_icon="🧪", layout="wide")
st.title("AI Reliability Experiment by Jack Eckel")
st.caption("Testing whether structured reasoning improves AI fact-checking reliability")

MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Experiment settings (keep constant for fairness) ---
TEMPERATURE = 0.35          
TOP_P = 1.0                
MAX_TOKENS_QUICK = 400     
MAX_TOKENS_REASON = 900    

# Keep visuals clean + readable (no seaborn dependency)
plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_QUICK = "#1f77b4"
COLOR_REASON = "#ff7f0e"

# ---------------------------------------------------------
# CLAIM DATASET
# ---------------------------------------------------------
CLAIMS = [
    {"text":"If Earth stopped rotating but continued orbiting the Sun, we would still have normal day and night cycles.","answer":"False"},
    {"text":"If you double the mass of an object in free fall in a vacuum, it will hit the ground at the same time as a lighter object.","answer":"True"},
    {"text":"Increasing the temperature of a gas in a sealed container increases the pressure.","answer":"True"},
    {"text":"If two objects have the same volume but different masses, they must have the same density.","answer":"False"},
    {"text":"An object moving at constant velocity has no net force acting on it.","answer":"True"},
    {"text":"If a plant is placed in the dark, it immediately stops cellular respiration.","answer":"False"},
    {"text":"In space, a rocket must continuously fire its engines to keep moving forward.","answer":"False"},
    {"text":"If two chemicals react and temperature decreases, the reaction cannot be exothermic.","answer":"True"},
    {"text":"If you increase the speed of a moving object, its kinetic energy increases linearly.","answer":"False"},
    {"text":"A metal object and a wooden object of the same mass will experience the same gravitational force near Earth’s surface.","answer":"True"},
    {"text":"If Earth had no atmosphere, the sky would appear blue during the day.","answer":"False"},
    {"text":"When ice melts, the total mass of the water remains the same.","answer":"True"},
    {"text":"If a chemical reaction absorbs heat from its surroundings, it is exothermic.","answer":"False"},
    {"text":"An object in orbit around Earth is constantly falling toward Earth.","answer":"True"},
    {"text":"If two planets have the same mass but different radii, the one with the smaller radius has stronger surface gravity.","answer":"True"},
    {"text":"If light travels through water instead of air, its wavelength stays exactly the same.","answer":"False"},
    {"text":"A higher pH value means a solution is more acidic.","answer":"False"},
    {"text":"Viruses are considered living organisms.","answer":"Uncertain"},
    {"text":"Pluto is a planet.","answer":"Uncertain"},
    {"text":"Glass is a solid.","answer":"Uncertain"},
]

# ---------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------
SYSTEM = "You are a careful science fact checker."

QUICK_PROMPT = """
If the claim depends on definitions, missing context, or could reasonably be debated, answer Uncertain.
Confidence should reflect scientific certainty, not how fluent the answer sounds.

Return the answer in this exact format:

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

REASON_PROMPT = """
Think step-by-step and explain reasoning using bullet points.
If the claim depends on definitions, missing context, or could reasonably be debated, answer Uncertain.
Confidence should reflect scientific certainty, not how fluent the answer sounds.

Return the answer in this exact format:

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
REASONING:
• <bullet 1>
• <bullet 2>
• <bullet 3>
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

# ---------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------
@dataclass
class Result:
    verdict: str
    confidence: int
    raw: str

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = {"quick": {}, "reason": {}}

if "judge_last_run" not in st.session_state:
    st.session_state.judge_last_run = []  # list of claim indices used in last judge run

if "claim_index" not in st.session_state:
    st.session_state.claim_index = 0

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def norm_label(x: str) -> str:
    x = (x or "").strip().lower()
    x = re.sub(r"[^\w]+$", "", x)  # strip trailing punctuation like "." or ")"
    if x == "true":
        return "True"
    if x == "false":
        return "False"
    return "Uncertain"

def badge(v: str) -> str:
    if v == "True":
        return "🟢 TRUE"
    if v == "False":
        return "🔴 FALSE"
    return "🟡 UNCERTAIN"

def parse(txt: str):
    v = re.search(r"VERDICT:\s*(True|False|Uncertain)", txt, re.I)
    c = re.search(r"CONFIDENCE:\s*(\d+)", txt)
    verdict = v.group(1).capitalize() if v else "Uncertain"
    confidence = int(c.group(1)) if c else 50
    return verdict, confidence

def ask_ai(claim: str, mode: str) -> Result:
    prompt = QUICK_PROMPT if mode == "quick" else REASON_PROMPT

    # Use different max_tokens for quick vs reasoning
    max_tokens = MAX_TOKENS_QUICK if mode == "quick" else MAX_TOKENS_REASON

    with st.spinner("Analyzing claim with AI..."):
        r = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt.format(claim=claim)},
            ],
        )

    txt = r.choices[0].message.content or ""
    v, c = parse(txt)
    return Result(v, c, txt)

def majority(results):
    votes = [r.verdict for r in results]
    return max(set(votes), key=votes.count)

def consistency(results):
    votes = [r.verdict for r in results]
    m = majority(results)
    return votes.count(m) / len(votes) * 100

def responsible_score(pred: str, truth: str) -> float:
    pred = norm_label(pred)
    truth = norm_label(truth)
    if pred == truth:
        return 1.0
    if pred == "Uncertain":
        return 0.5
    return 0.0

def mean_conf(results):
    return statistics.mean([x.confidence for x in results]) if results else 0.0

def small_bar(title: str, values):
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    bars = ax.bar(
        ["Quick", "Reasoning"],
        values,
        color=[COLOR_QUICK, COLOR_REASON],
        width=0.6
    )
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(100, max(values) * 1.15))
    st.pyplot(fig, clear_figure=True)

def section_card(title: str, body: str):
    st.markdown(
        f"""
<div style="padding:14px;border:1px solid rgba(0,0,0,.08);border-radius:14px;background:rgba(255,255,255,.6);">
  <div style="font-weight:700;font-size:16px;margin-bottom:6px;">{title}</div>
  <div style="font-size:14px;line-height:1.35;">{body}</div>
</div>
""",
        unsafe_allow_html=True,
    )

def clamp_index():
    if st.session_state.claim_index < 0:
        st.session_state.claim_index = 0
    if st.session_state.claim_index > len(CLAIMS) - 1:
        st.session_state.claim_index = len(CLAIMS) - 1

def reset_experiment():
    st.session_state.results = {"quick": {}, "reason": {}}
    st.session_state.judge_last_run = []
    st.session_state.claim_index = 0

# ---------------------------------------------------------
# ADDED: AI LIVE INTERPRETATION HELPER (NEW)
# ---------------------------------------------------------
def generate_ai_summary(metrics: dict) -> str:
    prompt = f"""
You are writing a clear, judge-friendly scientific interpretation for a middle-school AI experiment.

Two modes were compared:
- Quick Mode (answers immediately)
- Reasoning Mode (reasons step-by-step)

Use the metrics below to write 6–9 sentences:
1) What improved (or did not improve) and by how much
2) What uncertainty behavior suggests (human-like caution vs guessing)
3) What calibration suggests (confidence vs accuracy)
4) A simple takeaway sentence for judges

Metrics:
Claims tested: {metrics['n']}

Accuracy (%): Quick={metrics['acc_q']:.1f}, Reasoning={metrics['acc_r']:.1f}
Consistency (%): Quick={metrics['cons_q']:.1f}, Reasoning={metrics['cons_r']:.1f}
Avg Confidence (%): Quick={metrics['conf_q']:.1f}, Reasoning={metrics['conf_r']:.1f}
Uncertainty Rate (%): Quick={metrics['unc_q']:.1f}, Reasoning={metrics['unc_r']:.1f}
Responsible Answer Score: Quick={metrics['ras_q']:.1f}, Reasoning={metrics['ras_r']:.1f}
Overconfidence Rate (%): Quick={metrics['over_q']:.1f}, Reasoning={metrics['over_r']:.1f}

Calibration gaps (|confidence-accuracy|, lower is better):
Quick={metrics['cal_gap_q']:.1f}, Reasoning={metrics['cal_gap_r']:.1f}
"""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You analyze scientific experiment results clearly and conservatively."},
            {"role": "user", "content": prompt},
        ],
    )
    return r.choices[0].message.content

# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------
page = st.radio("", ["Instructions", "Experiment", "Ask Your Own Question", "Results"], horizontal=True)

# ---------------------------------------------------------
# INSTRUCTIONS PAGE
# ---------------------------------------------------------
if page == "Instructions":
    st.header("How to Run This Experiment")

    colA, colB = st.columns(2)
    with colA:
        section_card(
            "Goal",
            "Test whether <b>Reasoning Mode</b> produces more reliable science fact-checking than <b>Quick Mode</b>."
        )
        section_card(
            "Hypothesis",
            "If the AI reasons step-by-step, it will be <b>more accurate</b>, <b>more consistent</b>, and more willing to say <b>Uncertain</b> when evidence is unclear."
        )
    with colB:
        section_card(
            "How to Demo in 60 Seconds",
            "<ol>"
            "<li>Go to <b>Experiment</b>.</li>"
            "<li>Click <b>Run 5 Random Claims (Judge Demo)</b>.</li>"
            "<li>Point out Quick vs Reasoning differences in verdict, confidence, and reasoning bullets.</li>"
            "<li>Go to <b>Results</b> to show charts and the Responsible Answer Score.</li>"
            "</ol>"
        )

    st.subheader("What the Charts Mean")
    st.markdown("""
- **Accuracy**: Percent of claims where the verdict matches the known truth label.
- **Consistency**: If a claim is run 3 times, how often the AI returns the same verdict.
- **Average Confidence**: Average self-reported confidence (0–100).
- **Uncertainty Rate**: How often the AI says **Uncertain**.
- **Responsible Answer Score (RAS)**: Gives partial credit for **Uncertain** answers when the truth is True/False. This rewards “scientific caution.”
- **Overconfidence Rate**: For claims labeled **Uncertain**, measures how often the AI incorrectly answers True/False instead of admitting uncertainty.
""")

    st.info(
        "Key idea: In science, 'I’m not sure' can be the most responsible answer. "
        "This experiment measures not just accuracy, but also whether the AI uses uncertainty like a careful human scientist. "
        "Temperature controls randomness. "
        "In this experiment, temperature was held constant so the only independent variable was reasoning mode."
    )

# ---------------------------------------------------------
# EXPERIMENT PAGE
# ---------------------------------------------------------
elif page == "Experiment":
    clamp_index()

    top1, top2, top3 = st.columns([1, 1, 1])
    with top1:
        if st.button("Reset Experiment"):
            reset_experiment()
            st.success("Experiment reset.")

    with top2:
        run_three = st.checkbox("Run each claim 3 times (consistency test)", value=True)
        runs = 3 if run_three else 1

    with top3:
        st.caption(f"Runs per claim: {runs}")

    st.divider()

    # -----------------------
    # JUDGE DEMO MODE
    # -----------------------
    st.subheader("Judge Demo Mode (5 Random Claims)")
    st.caption("One-click demo for judges. Runs 5 random claims and shows Quick vs Reasoning.")

    if st.button("Run 5 Random Claims (Judge Demo)"):
        sample_idxs = random.sample(range(len(CLAIMS)), 5)
        st.session_state.judge_last_run = sample_idxs

        prog = st.progress(0)
        for k, i in enumerate(sample_idxs, start=1):
            claim = CLAIMS[i]["text"]
            st.session_state.results["quick"][i] = [ask_ai(claim, "quick") for _ in range(runs)]
            st.session_state.results["reason"][i] = [ask_ai(claim, "reason") for _ in range(runs)]
            prog.progress(k / 5)

    # Display judge results (always show if available)
    if st.session_state.judge_last_run:
        for i in st.session_state.judge_last_run:
            claim = CLAIMS[i]["text"]
            q = st.session_state.results["quick"].get(i)
            r = st.session_state.results["reason"].get(i)
            if not (q and r):
                continue

            st.markdown(f"### {claim}")

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### Quick Mode")
                st.write("Verdict:", badge(majority(q)))
                st.write("Confidence:", f"{mean_conf(q):.1f}%")
                if runs > 1:
                    st.write("Consistency:", f"{consistency(q):.1f}%")
                st.markdown(q[0].raw)

            with c2:
                st.markdown("#### Reasoning Mode")
                st.write("Verdict:", badge(majority(r)))          # ✅ verdict above bullets
                st.write("Confidence:", f"{mean_conf(r):.1f}%") # ✅ confidence above bullets
                if runs > 1:
                    st.write("Consistency:", f"{consistency(r):.1f}%")
                st.markdown(r[0].raw)

            st.divider()

    # -----------------------
    # MANUAL EXPERIMENT
    # -----------------------
    st.subheader("Manual Experiment (All 20 Claims)")
    idx = st.session_state.claim_index
    claim = CLAIMS[idx]["text"]

    st.caption(f"Claim {idx+1} / {len(CLAIMS)}")
    st.progress(idx / len(CLAIMS))
    st.write(f"**Claim:** {claim}")

    btn1, btn2, btn3 = st.columns([1, 1, 1])
    with btn1:
        if st.button("Run Quick (Manual)"):
            st.session_state.results["quick"][idx] = [ask_ai(claim, "quick") for _ in range(runs)]
    with btn2:
        if st.button("Run Reasoning (Manual)"):
            st.session_state.results["reason"][idx] = [ask_ai(claim, "reason") for _ in range(runs)]
    with btn3:
        if st.button("Next Claim ➜"):
            st.session_state.claim_index = min(len(CLAIMS) - 1, st.session_state.claim_index + 1)
            st.rerun()

    q = st.session_state.results["quick"].get(idx)
    r = st.session_state.results["reason"].get(idx)

    if q or r:
        left, right = st.columns(2)

        if q:
            with left:
                st.markdown("### Quick Mode")
                st.write("Verdict:", badge(majority(q)))
                st.write("Confidence:", f"{mean_conf(q):.1f}%")
                if runs > 1:
                    st.write("Consistency:", f"{consistency(q):.1f}%")
                st.markdown(q[0].raw)

        if r:
            with right:
                st.markdown("### Reasoning Mode")
                # ✅ verdict + confidence ABOVE bullets
                st.write("Verdict:", badge(majority(r)))
                st.write("Confidence:", f"{mean_conf(r):.1f}%")
                if runs > 1:
                    st.write("Consistency:", f"{consistency(r):.1f}%")
                st.markdown(r[0].raw)

# ---------------------------------------------------------
# ASK YOUR OWN QUESTION PAGE
# ---------------------------------------------------------
elif page == "Ask Your Own Question":
    st.header("Ask Your Own Question")
    st.caption("Enter a scientific claim and compare Quick Mode vs Reasoning Mode.")

    claim = st.text_input("Enter a scientific claim", placeholder="Example: A heavier object falls faster in a vacuum.")
    run_three = st.checkbox("Run 3 times (consistency test)", value=False)
    runs = 3 if run_three else 1

    if st.button("Analyze"):
        if not claim.strip():
            st.warning("Please enter a claim.")
        else:
            q = [ask_ai(claim, "quick") for _ in range(runs)]
            r = [ask_ai(claim, "reason") for _ in range(runs)]

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Quick Mode")
                st.write("Verdict:", badge(majority(q)))
                st.write("Confidence:", f"{mean_conf(q):.1f}%")
                if runs > 1:
                    st.write("Consistency:", f"{consistency(q):.1f}%")
                st.markdown(q[0].raw)

            with c2:
                st.subheader("Reasoning Mode")
                st.write("Verdict:", badge(majority(r)))
                st.write("Confidence:", f"{mean_conf(r):.1f}%")
                if runs > 1:
                    st.write("Consistency:", f"{consistency(r):.1f}%")
                st.markdown(r[0].raw)

# ---------------------------------------------------------
# RESULTS PAGE (CHARTS + EXPORT + REPORT)
# ---------------------------------------------------------
else:
    st.header("Results Dashboard")

    quick = st.session_state.results["quick"]
    reason = st.session_state.results["reason"]
    shared = sorted(set(quick) & set(reason))

    if not shared:
        st.warning("No completed comparisons yet. Run Judge Demo or Manual Experiment first.")
        st.stop()

    # Compute metrics
    acc_q = acc_r = 0
    cons_q = []
    cons_r = []
    unc_q = unc_r = 0
    ras_q = ras_r = 0.0
    unc_truth_total = 0
    over_q = 0
    over_r = 0

    processed = 0
    rows = []
    for i in shared:
        truth = norm_label(CLAIMS[i]["answer"])
        q_res = quick[i]
        r_res = reason[i]

        if not q_res or not r_res:
            continue
        processed += 1

        pred_q = norm_label(majority(q_res))
        pred_r = norm_label(majority(r_res))

        # Overconfidence (count only when truth is Uncertain)
        if truth == "Uncertain":
            unc_truth_total += 1
            if pred_q != "Uncertain":
                over_q += 1
            if pred_r != "Uncertain":
                over_r += 1

        # Accuracy
        if pred_q == truth:
            acc_q += 1
        if pred_r == truth:
            acc_r += 1

        # Uncertainty rate
        if pred_q == "Uncertain":
            unc_q += 1
        if pred_r == "Uncertain":
            unc_r += 1

        # Responsible Answer Score
        ras_q += responsible_score(pred_q, truth)
        ras_r += responsible_score(pred_r, truth)

        # Consistency
        cons_q.append(consistency(q_res))
        cons_r.append(consistency(r_res))

        # Row for table/export
        rows.append({
            "index": i + 1,
            "claim": CLAIMS[i]["text"],
            "truth": truth,
            "quick_verdict": pred_q,
            "quick_confidence_avg": round(mean_conf(q_res), 1),
            "reason_verdict": pred_r,
            "reason_confidence_avg": round(mean_conf(r_res), 1),
        })

    n = processed
    if n == 0:
        st.warning("No fully processed results yet. Run Judge Demo or Manual Experiment first.")
        st.stop()

    acc_q = acc_q / n * 100
    acc_r = acc_r / n * 100
    unc_q = unc_q / n * 100
    unc_r = unc_r / n * 100
    ras_q = ras_q / n * 100
    ras_r = ras_r / n * 100
    over_q = (over_q / unc_truth_total * 100) if unc_truth_total else 0.0
    over_r = (over_r / unc_truth_total * 100) if unc_truth_total else 0.0

    avg_conf_q = statistics.mean([rows[k]["quick_confidence_avg"] for k in range(len(rows))])
    avg_conf_r = statistics.mean([rows[k]["reason_confidence_avg"] for k in range(len(rows))])

    # calibration gaps (lower is better)
    cal_gap_q = abs(avg_conf_q - acc_q)
    cal_gap_r = abs(avg_conf_r - acc_r)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        section_card("Claims Tested", f"<b>{n}</b>")
    with k2:
        section_card("Accuracy Δ", f"<b>{(acc_r-acc_q):.1f} pts</b><br/><span style='font-size:12px;'>Reasoning - Quick</span>")
    with k3:
        section_card("RAS Δ", f"<b>{(ras_r-ras_q):.1f} pts</b><br/><span style='font-size:12px;'>Responsible Answer Score</span>")
    with k4:
        section_card(
            "Overconfidence Δ",
            f"<b>{(over_q - over_r):.1f} pts</b><br/><span style='font-size:12px;'>Lower is better</span>"
        )

    st.subheader("Charts")
    c1, c2, c3 = st.columns(3)
    with c1:
        small_bar("Accuracy (%)", [acc_q, acc_r])
    with c2:
        cons_title = "Consistency (%)" if n > 1 and (statistics.mean(cons_q) < 100 or statistics.mean(cons_r) < 100) else "Consistency (%) (1 run = 100%)"
        small_bar(cons_title, [statistics.mean(cons_q), statistics.mean(cons_r)])
    with c3:
        small_bar("Avg Confidence", [avg_conf_q, avg_conf_r])

    c4, c5, c6 = st.columns(3)
    with c4:
        small_bar("Uncertainty Rate (%)", [unc_q, unc_r])
    with c5:
        small_bar("Responsible Answer Score", [ras_q, ras_r])
    with c6:
        small_bar("Overconfidence Rate (%)", [over_q, over_r])

    st.subheader("Interpretation")
    st.write(
        "This experiment compares an AI that answers immediately (Quick) vs one that reasons step-by-step (Reasoning). "
        "In addition to accuracy, we measure whether the AI uses uncertainty responsibly. "
        "A higher **Responsible Answer Score** and lower **Overconfidence Rate** suggests the AI behaves more like a careful human scientist "
        "by saying **Uncertain** instead of guessing when information is unclear."
    )

    # ---------------------------------------------------------
    # ADDED: LIVE AI INTERPRETATION + REGENERATE BUTTON (NEW)
    # ---------------------------------------------------------
    metrics = {
        "n": n,
        "acc_q": acc_q,
        "acc_r": acc_r,
        "cons_q": statistics.mean(cons_q),
        "cons_r": statistics.mean(cons_r),
        "conf_q": avg_conf_q,
        "conf_r": avg_conf_r,
        "unc_q": unc_q,
        "unc_r": unc_r,
        "ras_q": ras_q,
        "ras_r": ras_r,
        "over_q": over_q,
        "over_r": over_r,
        "cal_gap_q": cal_gap_q,
        "cal_gap_r": cal_gap_r,
    }

    st.subheader("AI Live Interpretation")

    left_btn, right_btn = st.columns([3, 1])
    with right_btn:
        if st.button("Regenerate AI Interpretation"):
            st.session_state.pop("ai_summary", None)

    if "ai_summary" not in st.session_state:
        with st.spinner("Generating AI interpretation of the results..."):
            st.session_state.ai_summary = generate_ai_summary(metrics)

    st.write(st.session_state.ai_summary)

    # Data table + export
    df = pd.DataFrame(rows)
    st.subheader("Experiment Data")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results CSV",
        data=csv_bytes,
        file_name="ai_reliability_results.csv",
        mime="text/csv"
    )

    # Printable report (restored)
    st.subheader("Printable Report")
    report = f"""AI Reliability Experiment Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Claims tested: {n}

ACCURACY
- Quick: {acc_q:.1f}%
- Reasoning: {acc_r:.1f}%
- Difference (Reasoning - Quick): {(acc_r-acc_q):.1f} points

UNCERTAINTY BEHAVIOR
- Uncertainty Rate (Quick): {unc_q:.1f}%
- Uncertainty Rate (Reasoning): {unc_r:.1f}%

RESPONSIBLE ANSWERS
- Responsible Answer Score (Quick): {ras_q:.1f}
- Responsible Answer Score (Reasoning): {ras_r:.1f}

OVERCONFIDENCE (Lower is better)
- Overconfidence Rate (Quick): {over_q:.1f}%
- Overconfidence Rate (Reasoning): {over_r:.1f}%

CALIBRATION (Lower gap is better)
- |Confidence - Accuracy| Quick: {cal_gap_q:.1f}
- |Confidence - Accuracy| Reasoning: {cal_gap_r:.1f}

SETTINGS (CONTROLLED VARIABLES)
- Model: gpt-4o-mini
- Temperature: 0.2
- Top_p: 1.0
- Max tokens (Quick): 400
- Max tokens (Reasoning): 900

INTERPRETATION
Traditional accuracy counts 'Uncertain' as wrong, but in science, admitting uncertainty can be the most responsible response.
This report includes Responsible Answer Score and Overconfidence Rate to better capture careful scientific behavior.
"""
    st.download_button(
        "Download Printable Report (.txt)",
        data=report.encode("utf-8"),
        file_name="ai_reliability_report.txt",
        mime="text/plain"
    )