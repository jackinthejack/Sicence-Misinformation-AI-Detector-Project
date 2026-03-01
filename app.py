import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

# -----------------------------
# App Setup
# -----------------------------
st.set_page_config(
    page_title="Science Misinformation Detector (Experiment)",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Science Misinformation Detector")
st.caption(
    "6th Grade Science Fair Project by Jack Eckel: Does forcing AI to show reasoning improve accuracy at identifying misinformation?"
)

# Read API key from environment variable (Streamlit Cloud uses Secrets -> env var)
API_KEY = os.getenv("OPENAI_API_KEY", "")
if not API_KEY:
    st.warning(
        "OPENAI_API_KEY is not set yet. The app can run locally once you set it, or on Streamlit Cloud via Secrets."
    )

client = OpenAI(api_key=API_KEY) if API_KEY else None

MODEL_NAME = "gpt-4o-mini"  # cost-effective and fast


# -----------------------------
# Data: Preloaded Claims
# -----------------------------
# Keep claims age-appropriate and science-focused.
# answer must be one of: "True", "False"
PRELOADED_CLAIMS: List[Dict[str, str]] = [
    {"text": "The Earth orbits the Sun.", "answer": "True"},
    {"text": "Humans use only 10% of their brains.", "answer": "False"},
    {"text": "Water boils at 100°C (212°F) at sea level.", "answer": "True"},
    {"text": "Lightning never strikes the same place twice.", "answer": "False"},
    {"text": "Vaccines cause autism.", "answer": "False"},
    {"text": "Plants make their own food using photosynthesis.", "answer": "True"},
    {"text": "The Great Wall of China is visible from the Moon with the naked eye.", "answer": "False"},
    {"text": "Sound travels faster in water than in air.", "answer": "True"},
    {"text": "Seasons are caused because Earth is much closer to the Sun in summer.", "answer": "False"},
    {"text": "Antibiotics can kill viruses like the flu virus.", "answer": "False"},
    {"text": "All metals are attracted to magnets.", "answer": "False"},
    {"text": "Carbon dioxide is a greenhouse gas.", "answer": "True"},
    {"text": "The Moon produces its own light.", "answer": "False"},
    {"text": "A heavier object falls faster than a lighter one (in a vacuum).", "answer": "False"},
    {"text": "The ozone layer helps block some harmful ultraviolet (UV) radiation.", "answer": "True"},
    {"text": "Sugar always causes hyperactivity in children.", "answer": "False"},
    {"text": "Earth’s inner core is extremely hot.", "answer": "True"},
    {"text": "All bacteria are harmful to humans.", "answer": "False"},
    {"text": "A solar eclipse happens when the Moon blocks the Sun from view on Earth.", "answer": "True"},
    {"text": "Plastic decomposes quickly in nature (within a few years).", "answer": "False"},
]


# -----------------------------
# Prompts
# -----------------------------
SYSTEM_INSTRUCTIONS = """
You are a careful science fact-checking assistant for a school project.
Your job: judge whether a short science claim is TRUE, FALSE, or UNCERTAIN.
Be honest: if you do not have enough information, choose UNCERTAIN.
Use age-appropriate language for a 6th grader.
Avoid political content. Avoid medical advice. Provide general educational info only.
"""

QUICK_MODE_TEMPLATE = """
Evaluate this claim and respond in this exact format:

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>%
EXPLANATION: <2-4 sentences>

Claim: "{claim}"
"""

REASONING_MODE_TEMPLATE = """
Evaluate this claim carefully. First break it into parts, then decide.
Respond in this exact format:

REASONING:
- <bullet 1>
- <bullet 2>
- <bullet 3 (optional)>

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>%
EXPLANATION: <2-4 sentences>

Claim: "{claim}"
"""


# -----------------------------
# Helpers
# -----------------------------
@dataclass
class AIResult:
    verdict: str              # True|False|Uncertain
    confidence: int           # 0-100
    explanation: str
    raw_text: str
    seconds: float


def extract_result(text: str) -> Tuple[str, int, str]:
    """
    Parse model output. We enforce a strict format, but still parse defensively.
    """
    verdict = "Uncertain"
    confidence = 50
    explanation = ""

    # Verdict
    m = re.search(r"VERDICT:\s*(True|False|Uncertain)", text, re.IGNORECASE)
    if m:
        verdict = m.group(1).capitalize()

    # Confidence
    m = re.search(r"CONFIDENCE:\s*(\d{1,3})\s*%", text)
    if m:
        confidence = max(0, min(100, int(m.group(1))))

    # Explanation
    m = re.search(r"EXPLANATION:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        explanation = m.group(1).strip()
        # keep it from being too huge if the model rambles
        explanation = explanation[:800]

    return verdict, confidence, explanation


def call_ai(claim: str, mode: str) -> AIResult:
    if client is None:
        raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY.")

    prompt = (QUICK_MODE_TEMPLATE if mode == "quick" else REASONING_MODE_TEMPLATE).format(claim=claim)

    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS.strip()},
            {"role": "user", "content": prompt.strip()},
        ],
        temperature=0.2,
    )
    t1 = time.time()

    text = resp.choices[0].message.content or ""
    verdict, confidence, explanation = extract_result(text)

    return AIResult(
        verdict=verdict,
        confidence=confidence,
        explanation=explanation,
        raw_text=text,
        seconds=round(t1 - t0, 2),
    )


def badge(verdict: str) -> str:
    if verdict == "True":
        return "🟢 TRUE"
    if verdict == "False":
        return "🔴 FALSE"
    return "🟡 UNCERTAIN"


def compute_score(ai_verdict: str, correct_answer: str) -> bool:
    """
    Uncertain counts as incorrect (but we'll track it separately).
    """
    return ai_verdict in ("True", "False") and ai_verdict == correct_answer


def init_state():
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    # Store results per claim index
    # Example: st.session_state.results["quick"][idx] = {"verdict":..., "correct":..., ...}
    if "results" not in st.session_state:
        st.session_state.results = {"quick": {}, "reason": {}}

    if "view" not in st.session_state:
        st.session_state.view = "Experiment"

init_state()


# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.header("🧭 Navigation")
st.session_state.view = st.sidebar.radio("Choose a section:", ["Experiment", "Live Demo", "Results"])

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Project Variable")
st.sidebar.write("**Independent variable:** AI mode (Quick vs Reasoning)")
st.sidebar.write("**Dependent variable:** Accuracy (%)")

st.sidebar.markdown("---")
st.sidebar.subheader("🔐 Key Setup")
st.sidebar.write("On Streamlit Cloud, add `OPENAI_API_KEY` in **Secrets**.")


# -----------------------------
# Main Views
# -----------------------------
def render_experiment():
    st.subheader("🔬 Experiment Mode (Step-by-step)")

    total = len(PRELOADED_CLAIMS)
    idx = st.session_state.idx
    claim = PRELOADED_CLAIMS[idx]["text"]
    correct = PRELOADED_CLAIMS[idx]["answer"]

    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        st.markdown("### Claim")
        st.markdown(f"**{idx+1} of {total}**")
        st.info(claim)

        st.markdown("### Run Analysis")
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

        with btn_col1:
            if st.button("⚡ Run Quick Mode", use_container_width=True, disabled=(client is None)):
                res = call_ai(claim, mode="quick")
                st.session_state.results["quick"][idx] = {
                    "verdict": res.verdict,
                    "confidence": res.confidence,
                    "seconds": res.seconds,
                    "raw": res.raw_text,
                    "correct": compute_score(res.verdict, correct),
                    "uncertain": (res.verdict == "Uncertain"),
                }

        with btn_col2:
            if st.button("🧠 Run Reasoning Mode", use_container_width=True, disabled=(client is None)):
                res = call_ai(claim, mode="reason")
                st.session_state.results["reason"][idx] = {
                    "verdict": res.verdict,
                    "confidence": res.confidence,
                    "seconds": res.seconds,
                    "raw": res.raw_text,
                    "correct": compute_score(res.verdict, correct),
                    "uncertain": (res.verdict == "Uncertain"),
                }

        with btn_col3:
            if st.button("➡️ Next Claim", use_container_width=True):
                if st.session_state.idx < total - 1:
                    st.session_state.idx += 1
                else:
                    st.success("You reached the end! Go to the Results tab to see the chart.")

        st.caption("Tip: Run **both modes** on each claim before moving to the next claim.")

    with colB:
        st.markdown("### Answer Key (Hidden Option)")
        show_key = st.checkbox("Show correct label (for project owner)", value=False)
        if show_key:
            st.write(f"✅ Correct label: **{correct}**")
        else:
            st.write("🔒 Hidden during judging (recommended).")

        st.markdown("---")
        st.markdown("### Results for This Claim")

        q = st.session_state.results["quick"].get(idx)
        r = st.session_state.results["reason"].get(idx)

        def render_one(title: str, data: Optional[Dict]):
            st.markdown(f"**{title}**")
            if not data:
                st.write("— not run yet —")
                return
            st.write(badge(data["verdict"]))
            st.write(f"Confidence: **{data['confidence']}%**  |  Time: **{data['seconds']}s**")
            st.write(f"Scored correct? **{'Yes' if data['correct'] else 'No'}**")
            with st.expander("Show full AI output"):
                st.text(data["raw"])

        render_one("Quick Mode", q)
        st.markdown("---")
        render_one("Reasoning Mode", r)


def render_live_demo():
    st.subheader("🌎 Live Mode (User-entered claim)")
    st.write("Type any science claim.")

    claim = st.text_input("Enter a science claim:", placeholder="Example: 'Electricity is a type of energy.'")
    mode = st.radio("Choose AI mode:", ["Quick Mode", "Reasoning Mode"], horizontal=True)
    run = st.button("Analyze Claim", disabled=(client is None or not claim.strip()))

    if run:
        res = call_ai(claim.strip(), mode="quick" if mode.startswith("Quick") else "reason")
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.markdown("### Verdict")
            st.write(badge(res.verdict))
            st.write(f"Confidence: **{res.confidence}%**")
            st.write(f"Time: **{res.seconds}s**")
        with right:
            st.markdown("### Explanation")
            st.write(res.explanation)
            with st.expander("Show full AI output"):
                st.text(res.raw_text)


def render_results():
    st.subheader("📊 Results Dashboard")
    total = len(PRELOADED_CLAIMS)

    quick_results = st.session_state.results["quick"]
    reason_results = st.session_state.results["reason"]

    def summarize(results: Dict[int, Dict]) -> Tuple[int, int, int]:
        correct = sum(1 for _, d in results.items() if d.get("correct"))
        uncertain = sum(1 for _, d in results.items() if d.get("uncertain"))
        ran = len(results)
        return ran, correct, uncertain

    q_ran, q_correct, q_uncertain = summarize(quick_results)
    r_ran, r_correct, r_uncertain = summarize(reason_results)

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.metric("Claims Total", total)
    with col2:
        st.metric("Quick Mode Run", f"{q_ran}/{total}")
    with col3:
        st.metric("Reasoning Mode Run", f"{r_ran}/{total}")

    st.markdown("---")

    # Avoid division by zero
    q_acc = (q_correct / q_ran * 100) if q_ran else 0
    r_acc = (r_correct / r_ran * 100) if r_ran else 0

    colA, colB = st.columns([1, 1], gap="large")
    with colA:
        st.markdown("### Accuracy")
        st.write(f"⚡ Quick Mode Accuracy: **{q_acc:.1f}%**")
        st.write(f"🧠 Reasoning Mode Accuracy: **{r_acc:.1f}%**")

        fig = plt.figure()
        plt.bar(["Quick", "Reasoning"], [q_acc, r_acc])
        plt.ylim(0, 100)
        plt.ylabel("Accuracy (%)")
        st.pyplot(fig)

    with colB:
        st.markdown("### Uncertain Counts")
        st.write(f"⚡ Quick Mode Uncertain: **{q_uncertain}**")
        st.write(f"🧠 Reasoning Mode Uncertain: **{r_uncertain}**")

        fig2 = plt.figure()
        plt.bar(["Quick", "Reasoning"], [q_uncertain, r_uncertain])
        plt.ylabel("Uncertain Responses (count)")
        st.pyplot(fig2)

    st.markdown("---")
    st.markdown("### Reset / Start Over")
    if st.button("🔄 Reset Experiment Data"):
        st.session_state.idx = 0
        st.session_state.results = {"quick": {}, "reason": {}}
        st.success("Reset complete. Go back to Experiment mode to start again.")


# Render selected view
if st.session_state.view == "Experiment":
    render_experiment()
elif st.session_state.view == "Live Demo":
    render_live_demo()
else:
    render_results()