import os
import re
import csv
import random
from datetime import datetime
from dataclasses import dataclass

import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

st.set_page_config(page_title="AI Reliability Experiment by Jack Eckel", page_icon="🧪", layout="wide")

st.title("AI Reliability Experiment by Jack Eckel")
st.caption("Testing whether structured reasoning improves AI fact-checking reliability by Jack Eckel")

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_TOKENS = 800

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY) if API_KEY else None

# --------------------------------------------------
# DATASET
# --------------------------------------------------

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

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results={"quick":{}, "reason":{}}

# --------------------------------------------------
# REPRODUCIBILITY SEED
# --------------------------------------------------

seed = st.number_input("Random Seed (for reproducibility)", value=42)
random.seed(seed)

# --------------------------------------------------
# PROMPTS
# --------------------------------------------------

SYSTEM_PROMPT="You are a careful science fact checker."

QUICK_PROMPT="""
VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>%
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

REASON_PROMPT="""
REASONING:
Step 1: Identify key science concept
Step 2: Compare with known science

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>%
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

@dataclass
class AIResult:
    verdict:str
    confidence:int
    explanation:str
    raw:str

def parse(text):

    verdict="Uncertain"
    confidence=50
    explanation=""

    v=re.search(r"VERDICT:\s*(True|False|Uncertain)",text)
    if v: verdict=v.group(1)

    c=re.search(r"CONFIDENCE:\s*(\d+)",text)
    if c: confidence=int(c.group(1))

    e=re.search(r"EXPLANATION:\s*(.*)",text)
    if e: explanation=e.group(1)

    return verdict,confidence,explanation

def ask_ai(claim,mode):

    prompt=QUICK_PROMPT if mode=="quick" else REASON_PROMPT

    resp=client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":prompt.format(claim=claim)}
        ]
    )

    text=resp.choices[0].message.content
    verdict,conf,exp=parse(text)

    return AIResult(verdict,conf,exp,text)

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

page=st.radio("",["Experiment","Ask Your Own Question","Results"],horizontal=True)

# --------------------------------------------------
# EXPERIMENT PAGE
# --------------------------------------------------

if page=="Experiment":

    st.subheader("Experiment")
    st.info("Goal: Test whether structured reasoning improves AI reliability.")

    colA,colB=st.columns(2)

    if colA.button("Run 5 Random Claims (Judge Mode)"):

        indices=random.sample(range(len(CLAIMS)),5)

        for i in indices:

            claim=CLAIMS[i]["text"]

            quick=ask_ai(claim,"quick")
            reason=ask_ai(claim,"reason")

            st.session_state.results["quick"][i]=quick
            st.session_state.results["reason"][i]=reason

        st.success("Completed 5 random comparisons")

    if colB.button("Run Entire Experiment Automatically"):

        for i,c in enumerate(CLAIMS):

            claim=c["text"]

            quick=ask_ai(claim,"quick")
            reason=ask_ai(claim,"reason")

            st.session_state.results["quick"][i]=quick
            st.session_state.results["reason"][i]=reason

        st.success("Experiment complete")

# --------------------------------------------------
# ASK YOUR OWN QUESTION
# --------------------------------------------------

elif page=="Ask Your Own Question":

    st.subheader("Ask Your Own Question")

    claim=st.text_input("Enter a science claim")

    if st.button("Analyze"):

        quick=ask_ai(claim,"quick")
        reason=ask_ai(claim,"reason")

        c1,c2=st.columns(2)

        with c1:
            st.write("Quick Mode")
            st.write(quick.verdict)
            st.write(f"Confidence: {quick.confidence}%")
            st.write(quick.explanation)

        with c2:
            st.write("Reasoning Mode")
            st.write(reason.verdict)
            st.write(f"Confidence: {reason.confidence}%")
            st.write(reason.explanation)

# --------------------------------------------------
# RESULTS PAGE
# --------------------------------------------------

else:

    st.subheader("Results Dashboard")

    quick=st.session_state.results["quick"]
    reason=st.session_state.results["reason"]

    if not quick:
        st.warning("Run experiment first.")
        st.stop()

    acc_q=0
    acc_r=0
    conf_q=[]
    conf_r=[]

    for i in quick:

        truth=CLAIMS[i]["answer"]

        if quick[i].verdict==truth:
            acc_q+=1

        if reason[i].verdict==truth:
            acc_r+=1

        conf_q.append(quick[i].confidence)
        conf_r.append(reason[i].confidence)

    acc_q=acc_q/len(quick)*100
    acc_r=acc_r/len(reason)*100

    avg_q=sum(conf_q)/len(conf_q)
    avg_r=sum(conf_r)/len(conf_r)

    gap_q=abs(avg_q-acc_q)
    gap_r=abs(avg_r-acc_r)

    # --------------------------------------------------
    # DASHBOARD METRICS
    # --------------------------------------------------

    m1,m2,m3,m4=st.columns(4)

    m1.metric("Quick Accuracy",f"{acc_q:.1f}%")
    m2.metric("Reasoning Accuracy",f"{acc_r:.1f}%")
    m3.metric("Quick Calibration Gap",f"{gap_q:.1f}")
    m4.metric("Reasoning Calibration Gap",f"{gap_r:.1f}")

    st.divider()

    # --------------------------------------------------
    # CHARTS
    # --------------------------------------------------

    fig,ax=plt.subplots(figsize=(4,3))
    ax.bar(["Quick","Reason"],[acc_q,acc_r])
    ax.set_title("Accuracy")
    st.pyplot(fig)

    fig,ax=plt.subplots(figsize=(4,3))
    ax.bar(["Quick","Reason"],[avg_q,avg_r])
    ax.set_title("Average Confidence")
    st.pyplot(fig)

    fig,ax=plt.subplots(figsize=(4,3))
    ax.bar(["Quick","Reason"],[gap_q,gap_r])
    ax.set_title("Calibration Gap")
    st.pyplot(fig)

    st.markdown("### Results Summary")

    st.write(
        f"Reasoning mode achieved {acc_r:.1f}% accuracy compared to {acc_q:.1f}% in quick mode."
    )

    # --------------------------------------------------
    # CSV EXPORT
    # --------------------------------------------------

    rows=[]

    for i in quick:

        rows.append([
            datetime.now(),
            CLAIMS[i]["text"],
            CLAIMS[i]["answer"],
            quick[i].verdict,
            quick[i].confidence,
            reason[i].verdict,
            reason[i].confidence
        ])

    csv_file="results.csv"

    with open(csv_file,"w",newline="") as f:

        writer=csv.writer(f)

        writer.writerow([
            "timestamp",
            "claim",
            "ground_truth",
            "quick_verdict",
            "quick_confidence",
            "reason_verdict",
            "reason_confidence"
        ])

        writer.writerows(rows)

    with open(csv_file,"rb") as f:

        st.download_button("Download Experiment Data",f,"results.csv")

    if st.button("Reset Experiment"):

        st.session_state.results={"quick":{}, "reason":{}}
        st.rerun()