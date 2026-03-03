import os
import re
import time
from dataclasses import dataclass
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

st.set_page_config(page_title="AI Science Misinformation Detector", page_icon="🧪", layout="wide")

st.title("🧪 AI Science Misinformation Detector")
st.caption("Does requiring AI to show reasoning improve misinformation detection accuracy?")

API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY) if API_KEY else None
MODEL_NAME = "gpt-4o-mini"

# ---------------- Sidebar ----------------
st.sidebar.header("🔬 Research Settings")
temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.2)
max_tokens = st.sidebar.slider("Max Tokens", 200, 1200, 600)
SOFT_LIMIT = st.sidebar.number_input("Soft API Call Limit", 10, 200, 80)
COST_PER_CALL = 0.002

# ---------------- Session State ----------------
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "results" not in st.session_state:
    st.session_state.results = {"quick": {}, "reason": {}}
if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0

# ---------------- Harder Claims ----------------
CLAIMS = [
    {"text": "The Earth orbits the Sun.", "answer": "True"},
    {"text": "Humans use only 10% of their brains.", "answer": "False"},
    {"text": "Water boils at exactly 100°C everywhere on Earth.", "answer": "False"},
    {"text": "Lightning never strikes the same place twice.", "answer": "False"},
    {"text": "Cracking your knuckles causes arthritis.", "answer": "False"},
    {"text": "Plants get most of their mass from the soil.", "answer": "False"},
    {"text": "Sound travels faster in water than in air.", "answer": "True"},
    {"text": "Heavier objects fall faster than lighter objects in a vacuum.", "answer": "False"},
    {"text": "The Moon is larger than Pluto.", "answer": "True"},
    {"text": "Carbon dioxide is the most abundant gas in Earth's atmosphere.", "answer": "False"},
    {"text": "Seasons are caused by Earth being closer to the Sun in summer.", "answer": "False"},
    {"text": "A force is required to keep an object moving at constant velocity in space.", "answer": "False"},
]

# ---------------- Prompts ----------------
SYSTEM_PROMPT = "You are a careful science fact-checker for a 6th grade project."

QUICK_PROMPT = """
Respond EXACTLY in this format:
VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>%
EXPLANATION: <short explanation>
Claim: "{claim}"
"""

REASON_PROMPT = """
You MUST think step-by-step before deciding.

Respond EXACTLY in this format:
REASONING:
- Break the claim into parts.
- Identify scientific principles involved.
- Compare the claim to known evidence.
VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>%
EXPLANATION: <summary>
Claim: "{claim}"
"""

@dataclass
class AIResult:
    verdict: str
    confidence: int
    explanation: str
    raw_text: str

def extract(text):
    v = re.search(r"VERDICT:\s*(True|False|Uncertain)", text, re.I)
    c = re.search(r"CONFIDENCE:\s*(\d+)", text)
    e = re.search(r"EXPLANATION:\s*(.+)", text, re.I | re.S)
    return (
        v.group(1).capitalize() if v else "Uncertain",
        int(c.group(1)) if c else 50,
        e.group(1).strip() if e else ""
    )

def call_ai(claim, mode):
    if st.session_state.api_calls >= SOFT_LIMIT:
        st.warning("Soft call limit reached.")
        return AIResult("Uncertain", 0, "Limit reached.", "")
    prompt = QUICK_PROMPT if mode=="quick" else REASON_PROMPT
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":prompt.format(claim=claim)}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    st.session_state.api_calls += 1
    text = response.choices[0].message.content
    v,c,e = extract(text)
    return AIResult(v,c,e,text)

def badge(v):
    return {"True":"🟢 TRUE","False":"🔴 FALSE"}.get(v,"🟡 UNCERTAIN")

# ---------------- Experiment ----------------
st.subheader("🔬 Experiment Mode")

idx = st.session_state.idx
claim = CLAIMS[idx]["text"]
correct = CLAIMS[idx]["answer"]

progress = (idx+1)/len(CLAIMS)
st.progress(progress)

st.info(f"Claim {idx+1}/{len(CLAIMS)}: {claim}")

col1, col2 = st.columns(2)

if col1.button("Run Quick Mode"):
    st.session_state.results["quick"][idx] = call_ai(claim,"quick")

if col2.button("Run Reasoning Mode"):
    st.session_state.results["reason"][idx] = call_ai(claim,"reason")

st.markdown("---")

for mode in ["quick","reason"]:
    result = st.session_state.results[mode].get(idx)
    if result:
        st.write(f"### {mode.capitalize()} Mode")
        st.write(badge(result.verdict))
        st.write(f"Confidence: {result.confidence}%")
        st.write(f"Correct: {result.verdict==correct}")
        if "REASONING:" in result.raw_text:
            st.text(result.raw_text.split("VERDICT:")[0])
        st.write(result.explanation)
        st.markdown("---")

both_done = idx in st.session_state.results["quick"] and idx in st.session_state.results["reason"]

if st.button("Next Claim", disabled=not both_done):
    if idx < len(CLAIMS)-1:
        st.session_state.idx += 1
        st.rerun()

# ---------------- Live Demo ----------------
st.markdown("## 🌎 Live Demo (Side-by-Side Comparison)")
demo_claim = st.text_input("Enter a science claim for comparison:")

if st.button("Analyze Both Modes"):
    if demo_claim:
        quick = call_ai(demo_claim,"quick")
        reason = call_ai(demo_claim,"reason")

        colA, colB = st.columns(2)

        with colA:
            st.subheader("Quick Mode")
            st.write(badge(quick.verdict))
            st.write(f"Confidence: {quick.confidence}%")
            st.write(quick.explanation)

        with colB:
            st.subheader("Reasoning Mode")
            st.write(badge(reason.verdict))
            st.write(f"Confidence: {reason.confidence}%")
            if "REASONING:" in reason.raw_text:
                st.text(reason.raw_text.split("VERDICT:")[0])
            st.write(reason.explanation)

# ---------------- Results ----------------
st.markdown("## 📊 Results")

def summarize(mode):
    results = st.session_state.results[mode]
    correct = 0
    confidences = []
    for i,r in results.items():
        if r.verdict == CLAIMS[i]["answer"]:
            correct += 1
        confidences.append(r.confidence)
    acc = (correct/len(results)*100) if results else 0
    avg_conf = sum(confidences)/len(confidences) if confidences else 0
    return acc, avg_conf

quick_acc, quick_conf = summarize("quick")
reason_acc, reason_conf = summarize("reason")

st.write(f"Quick Accuracy: {quick_acc:.1f}%")
st.write(f"Reasoning Accuracy: {reason_acc:.1f}%")

fig = plt.figure()
plt.bar(["Quick","Reasoning"],[quick_acc,reason_acc])
plt.ylabel("Accuracy (%)")
st.pyplot(fig)

fig2 = plt.figure()
plt.bar(["Quick","Reasoning"],[quick_conf,reason_conf])
plt.ylabel("Average Confidence")
st.pyplot(fig2)