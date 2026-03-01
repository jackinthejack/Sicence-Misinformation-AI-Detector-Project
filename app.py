import os
import re
import time
from dataclasses import dataclass
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="AI Science Misinformation Detector",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 AI Science Misinformation Detector")
st.caption("Science Fair Project: Does requiring AI to show reasoning improve misinformation detection accuracy?")

# -----------------------------
# API Setup
# -----------------------------
API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY) if API_KEY else None
MODEL_NAME = "gpt-4o-mini"

# -----------------------------
# Research Settings (Sidebar)
# -----------------------------
st.sidebar.header("🔬 Research Settings")

temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.2)
max_tokens = st.sidebar.slider("Max Tokens", 200, 1200, 500)

SOFT_LIMIT = st.sidebar.number_input("Soft API Call Limit", 10, 200, 50)

COST_PER_CALL = 0.002  # rough estimate per call

# -----------------------------
# Session State
# -----------------------------
if "idx" not in st.session_state:
    st.session_state.idx = 0

if "results" not in st.session_state:
    st.session_state.results = {"quick": {}, "reason": {}}

if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0

if "quota_warning" not in st.session_state:
    st.session_state.quota_warning = False

if "view" not in st.session_state:
    st.session_state.view = "Experiment"

# -----------------------------
# Claims
# -----------------------------
CLAIMS = [
    {"text": "The Earth orbits the Sun.", "answer": "True"},
    {"text": "Humans use only 10% of their brains.", "answer": "False"},
    {"text": "Water boils at 100°C at sea level.", "answer": "True"},
    {"text": "Lightning never strikes the same place twice.", "answer": "False"},
    {"text": "Cracking your knuckles causes arthritis.", "answer": "False"},
    {"text": "Plants use photosynthesis to make food.", "answer": "True"},
    {"text": "The Moon produces its own light.", "answer": "False"},
    {"text": "Sound travels faster in water than in air.", "answer": "True"},
    {"text": "Heavier objects fall faster than lighter objects in a vacuum.", "answer": "False"},
    {"text": "Carbon dioxide is a greenhouse gas.", "answer": "True"},
]

# -----------------------------
# Prompts
# -----------------------------
SYSTEM_PROMPT = """
You are a careful science fact-checking assistant for a 6th grade science fair project.
Judge if a claim is TRUE, FALSE, or UNCERTAIN.
Use simple, clear language.
"""

QUICK_PROMPT = """
Respond EXACTLY in this format:

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>%
EXPLANATION: <2-3 sentence explanation>

Claim: "{claim}"
"""

REASON_PROMPT = """
You MUST think step-by-step before deciding.

Respond EXACTLY in this format:

REASONING:
- Step 1: Break the claim into smaller parts.
- Step 2: Identify scientific facts related to the claim.
- Step 3: Compare the claim to known scientific evidence.

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>%
EXPLANATION: <2-3 sentence summary of your conclusion>

Claim: "{claim}"
"""

# -----------------------------
# Helper Class
# -----------------------------
@dataclass
class AIResult:
    verdict: str
    confidence: int
    explanation: str
    raw_text: str
    seconds: float


# -----------------------------
# Helpers
# -----------------------------
def extract_result(text):
    verdict = "Uncertain"
    confidence = 50
    explanation = ""

    v = re.search(r"VERDICT:\s*(True|False|Uncertain)", text, re.I)
    if v:
        verdict = v.group(1).capitalize()

    c = re.search(r"CONFIDENCE:\s*(\d+)", text)
    if c:
        confidence = int(c.group(1))

    e = re.search(r"EXPLANATION:\s*(.+)", text, re.I | re.S)
    if e:
        explanation = e.group(1).strip()

    return verdict, confidence, explanation


def call_ai(claim, mode):
    if st.session_state.api_calls >= SOFT_LIMIT:
        st.warning("⚠️ Soft API call limit reached.")
        return AIResult("Uncertain", 0, "Call limit reached.", "", 0)

    try:
        prompt = QUICK_PROMPT if mode == "quick" else REASON_PROMPT
        formatted = prompt.format(claim=claim)

        start = time.time()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": formatted},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        end = time.time()

        st.session_state.api_calls += 1

        text = response.choices[0].message.content
        verdict, confidence, explanation = extract_result(text)

        return AIResult(verdict, confidence, explanation, text, round(end - start, 2))

    except Exception as e:
        if "429" in str(e):
            st.session_state.quota_warning = True
            st.error("⚠️ API quota exceeded.")
        else:
            st.error(f"Error: {e}")

        return AIResult("Uncertain", 0, "Error occurred.", "", 0)


def badge(verdict):
    if verdict == "True":
        return "🟢 TRUE"
    if verdict == "False":
        return "🔴 FALSE"
    return "🟡 UNCERTAIN"


# -----------------------------
# Usage Display
# -----------------------------
st.markdown("---")
st.subheader("📊 Session Usage")

cost_estimate = st.session_state.api_calls * COST_PER_CALL

st.write(f"API Calls: **{st.session_state.api_calls}**")
st.write(f"Estimated Cost: **${cost_estimate:.4f}**")

if st.session_state.quota_warning:
    st.warning("Quota warning triggered during this session.")

# -----------------------------
# Navigation
# -----------------------------
st.sidebar.markdown("---")
st.session_state.view = st.sidebar.radio("Navigation", ["Experiment", "Live Demo", "Results"])

# -----------------------------
# Experiment Mode
# -----------------------------
if st.session_state.view == "Experiment":
    st.subheader("🔬 Experiment Mode")

    idx = st.session_state.idx
    claim = CLAIMS[idx]["text"]
    correct = CLAIMS[idx]["answer"]

    st.info(f"Claim {idx+1}: {claim}")

    col1, col2 = st.columns(2)

    if col1.button("⚡ Run Quick Mode"):
        result = call_ai(claim, "quick")
        st.session_state.results["quick"][idx] = result

    if col2.button("🧠 Run Reasoning Mode"):
        result = call_ai(claim, "reason")
        st.session_state.results["reason"][idx] = result

    st.markdown("---")

    for mode in ["quick", "reason"]:
        result = st.session_state.results[mode].get(idx)
        if result:
            st.write(f"### {mode.capitalize()} Mode")
            st.write(badge(result.verdict))
            st.write(f"Confidence: {result.confidence}%")
            st.write(f"Correct: {result.verdict == correct}")
            st.write(f"Time: {result.seconds}s")

            if "REASONING:" in result.raw_text:
                reasoning_section = result.raw_text.split("VERDICT:")[0]
                st.markdown("**Reasoning Steps:**")
                st.text(reasoning_section)

            st.markdown("**Final Explanation:**")
            st.write(result.explanation)
            st.markdown("---")

    if st.button("Next Claim"):
        if idx < len(CLAIMS) - 1:
            st.session_state.idx += 1
        else:
            st.success("Experiment complete! View Results.")

# -----------------------------
# Live Demo
# -----------------------------
elif st.session_state.view == "Live Demo":
    st.subheader("🌎 Live Demo Mode")

    claim = st.text_input("Enter a science claim:")
    mode = st.radio("Mode", ["Quick Mode", "Reasoning Mode"])

    if st.button("Analyze"):
        if claim:
            result = call_ai(claim, "quick" if mode == "Quick Mode" else "reason")
            st.write(badge(result.verdict))
            st.write(f"Confidence: {result.confidence}%")
            st.write(result.explanation)

# -----------------------------
# Results
# -----------------------------
else:
    st.subheader("📊 Results Dashboard")

    def summarize(mode):
        results = st.session_state.results[mode]
        correct = 0
        confidences = []

        for idx, result in results.items():
            answer = CLAIMS[idx]["answer"]
            if result.verdict == answer:
                correct += 1
            confidences.append(result.confidence)

        accuracy = (correct / len(results) * 100) if results else 0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        return accuracy, avg_conf

    quick_acc, quick_conf = summarize("quick")
    reason_acc, reason_conf = summarize("reason")

    st.write(f"Quick Accuracy: {quick_acc:.1f}%")
    st.write(f"Reasoning Accuracy: {reason_acc:.1f}%")

    fig = plt.figure()
    plt.bar(["Quick", "Reasoning"], [quick_acc, reason_acc])
    plt.ylabel("Accuracy (%)")
    st.pyplot(fig)

    st.markdown("---")
    st.write("### Average Confidence")

    fig2 = plt.figure()
    plt.bar(["Quick", "Reasoning"], [quick_conf, reason_conf])
    plt.ylabel("Average Confidence (%)")
    st.pyplot(fig2)

    if st.button("Reset Experiment"):
        st.session_state.idx = 0
        st.session_state.results = {"quick": {}, "reason": {}}
        st.session_state.api_calls = 0
        st.session_state.quota_warning = False
        st.success("Experiment reset.")