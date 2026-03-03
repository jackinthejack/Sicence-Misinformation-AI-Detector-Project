# (shortened explanation header removed for clarity)

import os
import re
import csv
import io
import random
import time
from dataclasses import dataclass
from datetime import datetime

import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

st.set_page_config(page_title="AI Reliability Experiment", page_icon="🧪", layout="wide")

st.title("AI Reliability Experiment")
st.caption("Does structured reasoning improve AI fact-checking reliability?")

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_TOKENS = 800

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY) if API_KEY else None


# --------------------------
# Dataset
# --------------------------

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

# --------------------------
# Session State
# --------------------------

if "results" not in st.session_state:
    st.session_state.results={"quick":{}, "reason":{}}

if "idx" not in st.session_state:
    st.session_state.idx=0

if "batch" not in st.session_state:
    st.session_state.batch=None

if "ask_result" not in st.session_state:
    st.session_state.ask_result=None


# --------------------------
# AI Prompts
# --------------------------

SYSTEM="You are a careful science fact-checker."

QUICK="""
VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

REASON="""
REASONING:
Step 1: Identify scientific principle
Step 2: Compare to known science

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
EXPLANATION: <short explanation>

Claim: "{claim}"
"""


@dataclass
class Result:
    verdict:str
    confidence:int
    explanation:str
    raw:str


# --------------------------
# Helpers
# --------------------------

def parse(text):

    v=re.search("VERDICT:\s*(True|False|Uncertain)",text)
    c=re.search("CONFIDENCE:\s*(\d+)",text)
    e=re.search("EXPLANATION:\s*(.*)",text)

    verdict=v.group(1) if v else "Uncertain"
    conf=int(c.group(1)) if c else 50
    exp=e.group(1) if e else ""

    return verdict,conf,exp


def ask_ai(claim,mode):

    prompt=QUICK if mode=="quick" else REASON

    with st.spinner("🔄 AI analyzing claim..."):

        resp=client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role":"system","content":SYSTEM},
                {"role":"user","content":prompt.format(claim=claim)}
            ]
        )

    txt=resp.choices[0].message.content

    v,c,e=parse(txt)

    return Result(v,c,e,txt)


def badge(v):

    if v=="True": return "🟢 TRUE"
    if v=="False": return "🔴 FALSE"
    return "🟡 UNCERTAIN"


# --------------------------
# Navigation
# --------------------------

page=st.radio("",["Experiment","Ask Your Own Question","Results"],horizontal=True)


# ====================================================
# EXPERIMENT
# ====================================================

if page=="Experiment":

    st.subheader("Experiment")

    progress=len(set(st.session_state.results["quick"]) & set(st.session_state.results["reason"]))

    st.progress(progress/20)
    st.write(f"Progress: {progress}/20 claims completed")

    col1,col2=st.columns(2)

    if col1.button("Run 5 Random Claims (Judge Demo)"):

        batch=[]

        for i in random.sample(range(20),5):

            claim=CLAIMS[i]["text"]
            truth=CLAIMS[i]["answer"]

            q=ask_ai(claim,"quick")
            r=ask_ai(claim,"reason")

            st.session_state.results["quick"][i]=q
            st.session_state.results["reason"][i]=r

            batch.append((i,claim,truth,q,r))

        st.session_state.batch=batch

    if col2.button("Restart Manual Experiment"):
        st.session_state.idx=0
        st.session_state.batch=None


    if st.session_state.batch:

        st.subheader("Judge Demo Results")

        for i,claim,truth,q,r in st.session_state.batch:

            st.write("###",claim)

            c1,c2=st.columns(2)

            with c1:
                st.write("Quick")
                st.write(badge(q.verdict),q.confidence)
                st.write(q.explanation)

            with c2:
                st.write("Reasoning")
                st.write(badge(r.verdict),r.confidence)
                st.write(r.explanation)

            st.divider()


    idx=st.session_state.idx
    claim=CLAIMS[idx]["text"]

    st.write("### Manual Claim",idx+1)
    st.write(claim)

    if st.button("Run Quick Mode"):
        st.session_state.results["quick"][idx]=ask_ai(claim,"quick")
        st.rerun()

    if st.button("Run Reasoning Mode"):
        st.session_state.results["reason"][idx]=ask_ai(claim,"reason")
        st.rerun()

    q=st.session_state.results["quick"].get(idx)
    r=st.session_state.results["reason"].get(idx)

    if q:
        st.write("Quick:",badge(q.verdict),q.confidence)

    if r:
        st.write("Reason:",badge(r.verdict),r.confidence)

    if q and r:

        if st.button("Next Claim"):

            if idx<19:
                st.session_state.idx+=1
                st.rerun()
            else:
                st.success("Experiment complete — view Results.")


# ====================================================
# ASK QUESTION
# ====================================================

elif page=="Ask Your Own Question":

    claim=st.text_input("Enter a science claim")

    if st.button("Analyze"):

        q=ask_ai(claim,"quick")
        r=ask_ai(claim,"reason")

        st.session_state.ask_result=(claim,q,r)

    if st.session_state.ask_result:

        claim,q,r=st.session_state.ask_result

        c1,c2=st.columns(2)

        with c1:
            st.write("Quick Mode")
            st.write(badge(q.verdict),q.confidence)
            st.write(q.explanation)

        with c2:
            st.write("Reasoning Mode")
            st.write(badge(r.verdict),r.confidence)
            st.write(r.explanation)


# ====================================================
# RESULTS
# ====================================================

else:

    quick=st.session_state.results["quick"]
    reason=st.session_state.results["reason"]

    shared=set(quick)&set(reason)

    if not shared:
        st.warning("Run experiment first")
        st.stop()

    correct_q=0
    correct_r=0

    conf_q=[]
    conf_r=[]

    for i in shared:

        truth=CLAIMS[i]["answer"]

        if quick[i].verdict==truth:
            correct_q+=1

        if reason[i].verdict==truth:
            correct_r+=1

        conf_q.append(quick[i].confidence)
        conf_r.append(reason[i].confidence)

    n=len(shared)

    acc_q=correct_q/n*100
    acc_r=correct_r/n*100

    avg_q=sum(conf_q)/n
    avg_r=sum(conf_r)/n

    st.subheader("Experiment Results")

    if acc_r>acc_q:
        st.success(f"🏆 Reasoning Mode performed better (+{acc_r-acc_q:.1f}% accuracy)")
    else:
        st.info("Quick Mode performed similarly")

    col1,col2=st.columns(2)

    with col1:
        st.metric("Quick Accuracy",f"{acc_q:.1f}%")

    with col2:
        st.metric("Reason Accuracy",f"{acc_r:.1f}%")

    fig,ax=plt.subplots()

    ax.bar(["Quick","Reason"],[acc_q,acc_r])

    st.pyplot(fig)

    csv_buffer=io.StringIO()
    writer=csv.writer(csv_buffer)

    writer.writerow(["claim","truth","quick","reason"])

    for i in shared:

        writer.writerow([
            CLAIMS[i]["text"],
            CLAIMS[i]["answer"],
            quick[i].verdict,
            reason[i].verdict
        ])

    st.download_button(
        "Download Experiment Data",
        csv_buffer.getvalue(),
        "experiment_results.csv"
    )