import os
import re
import csv
import io
import random
from dataclasses import dataclass
from datetime import datetime

import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

st.set_page_config(page_title="AI Reliability Experiment by Jack Eckel", page_icon="🧪", layout="wide")

st.title("AI Reliability Experiment by Jack Eckel")
st.caption("Testing whether structured reasoning improves AI fact-checking reliability by Jack Eckel")

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_TOKENS = 800

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# CLAIM DATASET
# ---------------------------

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

# ---------------------------
# SESSION STATE
# ---------------------------

if "results" not in st.session_state:
    st.session_state.results = {"quick":{}, "reason":{}}

if "idx" not in st.session_state:
    st.session_state.idx = 0

if "batch" not in st.session_state:
    st.session_state.batch = None

if "ask" not in st.session_state:
    st.session_state.ask = None

# ---------------------------
# PROMPTS
# ---------------------------

SYSTEM = "You are a careful science fact-checker."

QUICK_PROMPT = """
VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

REASON_PROMPT = """
REASONING:
Step 1: Identify the scientific concept
Step 2: Compare with established science

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

# ---------------------------
# RESULT STRUCT
# ---------------------------

@dataclass
class Result:
    verdict:str
    confidence:int
    explanation:str
    raw:str

# ---------------------------
# PARSER
# ---------------------------

def parse(text):

    v = re.search(r"VERDICT:\s*(True|False|Uncertain)", text)
    c = re.search(r"CONFIDENCE:\s*(\d+)", text)
    e = re.search(r"EXPLANATION:\s*(.*)", text)

    verdict = v.group(1) if v else "Uncertain"
    confidence = int(c.group(1)) if c else 50
    explanation = e.group(1) if e else ""

    return verdict, confidence, explanation

# ---------------------------
# OPENAI CALL
# ---------------------------

def ask_ai(claim, mode):

    prompt = QUICK_PROMPT if mode=="quick" else REASON_PROMPT

    with st.spinner("AI analyzing..."):

        r = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role":"system","content":SYSTEM},
                {"role":"user","content":prompt.format(claim=claim)}
            ]
        )

    txt = r.choices[0].message.content

    v,c,e = parse(txt)

    return Result(v,c,e,txt)

# ---------------------------
# HELPERS
# ---------------------------

def badge(v):

    if v=="True":
        return "🟢 TRUE"

    if v=="False":
        return "🔴 FALSE"

    return "🟡 UNCERTAIN"

def correct(v,t):
    return v == t

# ---------------------------
# NAVIGATION
# ---------------------------

page = st.radio("",["Experiment","Ask Your Own Question","Results"],horizontal=True)

# ==========================================================
# EXPERIMENT
# ==========================================================

if page=="Experiment":

    completed = len(set(st.session_state.results["quick"]) &
                    set(st.session_state.results["reason"]))

    st.progress(completed/20)

    st.write(f"Progress: {completed}/20 claims completed")

    col1,col2 = st.columns(2)

    if col1.button("Run 5 Random Claims (Judge Demo)"):

        batch=[]

        for i in random.sample(range(20),5):

            claim = CLAIMS[i]["text"]
            truth = CLAIMS[i]["answer"]

            q = ask_ai(claim,"quick")
            r = ask_ai(claim,"reason")

            st.session_state.results["quick"][i] = q
            st.session_state.results["reason"][i] = r

            batch.append((i,claim,truth,q,r))

        st.session_state.batch=batch

    if col2.button("Restart Experiment"):

        st.session_state.idx = 0
        st.session_state.batch = None
        st.session_state.results={"quick":{}, "reason":{}}

    if st.session_state.batch:

        st.subheader("Judge Demo Results")

        for i,claim,truth,q,r in st.session_state.batch:

            st.write("###",claim)

            c1,c2 = st.columns(2)

            with c1:
                st.write("Quick Mode")
                st.write(badge(q.verdict),q.confidence,"%")
                st.write("Correct:",correct(q.verdict,truth))
                st.write(q.explanation)

            with c2:
                st.write("Reasoning Mode")
                st.write(badge(r.verdict),r.confidence,"%")
                st.write("Correct:",correct(r.verdict,truth))
                st.write(r.raw)

            st.divider()

    idx = st.session_state.idx

    claim = CLAIMS[idx]["text"]
    truth = CLAIMS[idx]["answer"]

    st.subheader(f"Manual Claim {idx+1}/20")

    st.write(claim)

    c1,c2 = st.columns(2)

    if c1.button("Run Quick Mode"):
        st.session_state.results["quick"][idx]=ask_ai(claim,"quick")
        st.rerun()

    if c2.button("Run Reasoning Mode"):
        st.session_state.results["reason"][idx]=ask_ai(claim,"reason")
        st.rerun()

    q = st.session_state.results["quick"].get(idx)
    r = st.session_state.results["reason"].get(idx)

    if q:
        st.write("Quick:",badge(q.verdict),q.confidence,"%")
        st.write(q.explanation)

    if r:
        st.write("Reason:",badge(r.verdict),r.confidence,"%")
        st.write(r.raw)

    if q and r:

        if st.button("Next Claim"):

            if idx < 19:
                st.session_state.idx += 1
                st.rerun()
            else:
                st.success("Experiment complete — view Results")

# ==========================================================
# ASK YOUR OWN QUESTION
# ==========================================================

elif page=="Ask Your Own Question":

    claim = st.text_input("Enter a science claim")

    if st.button("Analyze"):

        q = ask_ai(claim,"quick")
        r = ask_ai(claim,"reason")

        st.session_state.ask = (claim,q,r)

    if st.session_state.ask:

        claim,q,r = st.session_state.ask

        c1,c2 = st.columns(2)

        with c1:
            st.write("Quick Mode")
            st.write(badge(q.verdict),q.confidence,"%")
            st.write(q.explanation)

        with c2:
            st.write("Reasoning Mode")
            st.write(badge(r.verdict),r.confidence,"%")
            st.write(r.raw)

# ==========================================================
# RESULTS DASHBOARD
# ==========================================================

else:

    quick = st.session_state.results["quick"]
    reason = st.session_state.results["reason"]

    shared = set(quick) & set(reason)

    if not shared:
        st.warning("Run experiment first.")
        st.stop()

    correct_q=0
    correct_r=0

    conf_q=[]
    conf_r=[]

    unc_q=0
    unc_r=0

    for i in shared:

        truth=CLAIMS[i]["answer"]

        q=quick[i]
        r=reason[i]

        if q.verdict==truth:
            correct_q+=1

        if r.verdict==truth:
            correct_r+=1

        conf_q.append(q.confidence)
        conf_r.append(r.confidence)

        if q.verdict=="Uncertain":
            unc_q+=1

        if r.verdict=="Uncertain":
            unc_r+=1

    n=len(shared)

    acc_q=correct_q/n*100
    acc_r=correct_r/n*100

    avg_q=sum(conf_q)/n
    avg_r=sum(conf_r)/n

    gap_q=abs(avg_q-acc_q)
    gap_r=abs(avg_r-acc_r)

    unc_q=unc_q/n*100
    unc_r=unc_r/n*100

    st.subheader("Experiment Results")

    m1,m2,m3,m4 = st.columns(4)

    m1.metric("Quick Accuracy",f"{acc_q:.1f}%")
    m2.metric("Reasoning Accuracy",f"{acc_r:.1f}%")
    m3.metric("Quick Calibration Gap",f"{gap_q:.1f}")
    m4.metric("Reasoning Calibration Gap",f"{gap_r:.1f}")

    st.divider()

    def chart(title,labels,values,y):

        fig,ax = plt.subplots(figsize=(4.5,3))

        bars=ax.bar(labels,values)

        for b in bars:
            ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{b.get_height():.1f}",ha='center')

        ax.set_title(title)
        ax.set_ylabel(y)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig)

    col1,col2 = st.columns(2)

    with col1:
        chart("Accuracy",["Quick","Reasoning"],[acc_q,acc_r],"Percent")

    with col2:
        chart("Average Confidence",["Quick","Reasoning"],[avg_q,avg_r],"Percent")

    col3,col4 = st.columns(2)

    with col3:
        chart("Calibration Gap",["Quick","Reasoning"],[gap_q,gap_r],"Gap")

    with col4:
        chart("Uncertainty Rate",["Quick","Reasoning"],[unc_q,unc_r],"Percent")

    # CSV export

    csv_buffer=io.StringIO()
    writer=csv.writer(csv_buffer)

    writer.writerow(["claim","truth","quick_verdict","quick_conf","reason_verdict","reason_conf"])

    for i in shared:

        writer.writerow([
            CLAIMS[i]["text"],
            CLAIMS[i]["answer"],
            quick[i].verdict,
            quick[i].confidence,
            reason[i].verdict,
            reason[i].confidence
        ])

    st.download_button(
        "Download Experiment Data",
        csv_buffer.getvalue(),
        "experiment_results.csv"
    )