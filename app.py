import os
import re
import csv
import io
import random
import statistics
from dataclasses import dataclass

import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

st.set_page_config(page_title="AI Reliability Experiment", page_icon="🧪", layout="wide")

st.title("AI Reliability Experiment")

MODEL="gpt-4o-mini"
TEMPERATURE=0.2
MAX_TOKENS=800

client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------
# CLAIM DATASET
# -------------------------------------------------------

CLAIMS=[
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

# -------------------------------------------------------
# PROMPTS
# -------------------------------------------------------

SYSTEM="You are a careful science fact checker."

QUICK="""
VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

REASON="""
REASONING:
Step 1: Identify scientific principle
Step 2: Compare with established science

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

# -------------------------------------------------------
# DATA STRUCTURES
# -------------------------------------------------------

@dataclass
class Result:
    verdict:str
    confidence:int
    explanation:str
    raw:str

# -------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results={"quick":{}, "reason":{}}

if "idx" not in st.session_state:
    st.session_state.idx=0

if "ask" not in st.session_state:
    st.session_state.ask=None

# -------------------------------------------------------
# PARSER
# -------------------------------------------------------

def parse(txt):

    v=re.search(r"VERDICT:\s*(True|False|Uncertain)",txt)
    c=re.search(r"CONFIDENCE:\s*(\d+)",txt)
    e=re.search(r"EXPLANATION:\s*(.*)",txt)

    verdict=v.group(1) if v else "Uncertain"
    conf=int(c.group(1)) if c else 50
    exp=e.group(1) if e else ""

    return verdict,conf,exp

# -------------------------------------------------------
# OPENAI CALL
# -------------------------------------------------------

def ask_ai(claim,mode):

    prompt=QUICK if mode=="quick" else REASON

    with st.spinner("AI analyzing..."):

        r=client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role":"system","content":SYSTEM},
                {"role":"user","content":prompt.format(claim=claim)}
            ]
        )

    txt=r.choices[0].message.content

    v,c,e=parse(txt)

    return Result(v,c,e,txt)

# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------

def badge(v):

    if v=="True":
        return "🟢 TRUE"
    if v=="False":
        return "🔴 FALSE"
    return "🟡 UNCERTAIN"

def majority_verdict(results):

    votes=[r.verdict for r in results]
    return max(set(votes),key=votes.count)

def consistency(results):

    votes=[r.verdict for r in results]
    majority=max(set(votes),key=votes.count)
    return votes.count(majority)/len(votes)*100

# -------------------------------------------------------
# NAVIGATION
# -------------------------------------------------------

page=st.radio("",["How This Experiment Works","Experiment","Ask Your Own Question","Results"],horizontal=True)

# =====================================================
# HOW THIS EXPERIMENT WORKS
# =====================================================

if page=="How This Experiment Works":

    st.header("How This Experiment Works")

    st.subheader("Goal")

    st.write("""
This project tests whether **structured reasoning improves the reliability of AI fact-checking**.
""")

    st.subheader("Hypothesis")

    st.write("""
If the AI performs step-by-step reasoning before answering, it will be **more accurate and more consistent**.
""")

    st.subheader("Variables")

    st.write("""
Independent Variable: AI response mode  
Dependent Variables:
- Accuracy
- Confidence
- Calibration Gap
- Uncertainty Rate
- Consistency
- Confidence Variability
""")

    st.subheader("Why each claim runs 3 times")

    st.write("""
AI models include randomness. Running each claim multiple times allows us to measure **answer stability**.
""")

    st.subheader("How to test the app")

    st.write("""
1. Run the experiment  
2. Compare Quick vs Reasoning mode  
3. View the Results dashboard  
""")

# =====================================================
# EXPERIMENT
# =====================================================

elif page=="Experiment":

    st.header("Experiment")

    multi_run=st.checkbox("Run each claim 3 times (consistency test)",value=True)

    runs=3 if multi_run else 1

    idx=st.session_state.idx

    claim=CLAIMS[idx]["text"]
    truth=CLAIMS[idx]["answer"]

    st.write(f"Claim {idx+1}/20")
    st.write(claim)

    if st.button("Run Quick Mode"):

        results=[ask_ai(claim,"quick") for _ in range(runs)]
        st.session_state.results["quick"][idx]=results
        st.rerun()

    if st.button("Run Reasoning Mode"):

        results=[ask_ai(claim,"reason") for _ in range(runs)]
        st.session_state.results["reason"][idx]=results
        st.rerun()

    q=st.session_state.results["quick"].get(idx)
    r=st.session_state.results["reason"].get(idx)

    if q:

        mv=majority_verdict(q)
        conf=sum([x.confidence for x in q])/len(q)

        st.write("Quick Result:",badge(mv),round(conf,1),"%")
        st.write("Consistency:",round(consistency(q),1),"%")

    if r:

        mv=majority_verdict(r)
        conf=sum([x.confidence for x in r])/len(r)

        st.write("Reason Result:",badge(mv),round(conf,1),"%")
        st.write("Consistency:",round(consistency(r),1),"%")

    if q and r:

        if st.button("Next Claim"):

            if idx<19:
                st.session_state.idx+=1
                st.rerun()

# =====================================================
# ASK PAGE
# =====================================================

elif page=="Ask Your Own Question":

    claim=st.text_input("Enter a science claim")

    if st.button("Analyze"):

        q=ask_ai(claim,"quick")
        r=ask_ai(claim,"reason")

        st.session_state.ask=(q,r)

    if st.session_state.ask:

        q,r=st.session_state.ask

        c1,c2=st.columns(2)

        with c1:
            st.write("Quick")
            st.write(badge(q.verdict),q.confidence,"%")

        with c2:
            st.write("Reasoning")
            st.write(badge(r.verdict),r.confidence,"%")

# =====================================================
# RESULTS DASHBOARD
# =====================================================

else:

    quick=st.session_state.results["quick"]
    reason=st.session_state.results["reason"]

    shared=set(quick)&set(reason)

    if not shared:
        st.warning("Run experiment first")
        st.stop()

    acc_q=0
    acc_r=0

    cons_q=[]
    cons_r=[]

    conf_q=[]
    conf_r=[]

    for i in shared:

        truth=CLAIMS[i]["answer"]

        q=quick[i]
        r=reason[i]

        mv_q=majority_verdict(q)
        mv_r=majority_verdict(r)

        if mv_q==truth:
            acc_q+=1

        if mv_r==truth:
            acc_r+=1

        cons_q.append(consistency(q))
        cons_r.append(consistency(r))

        conf_q.append(sum([x.confidence for x in q])/len(q))
        conf_r.append(sum([x.confidence for x in r])/len(r))

    n=len(shared)

    acc_q=acc_q/n*100
    acc_r=acc_r/n*100

    st.header("Results Dashboard")

    m1,m2=st.columns(2)

    m1.metric("Quick Accuracy",f"{acc_q:.1f}%")
    m2.metric("Reasoning Accuracy",f"{acc_r:.1f}%")

    def chart(title,labels,values):

        fig,ax=plt.subplots()

        bars=ax.bar(labels,values)

        for b in bars:
            ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{b.get_height():.1f}",ha='center')

        ax.set_title(title)

        st.pyplot(fig)

    chart("Accuracy",["Quick","Reasoning"],[acc_q,acc_r])
    chart("Consistency",["Quick","Reasoning"],[sum(cons_q)/n,sum(cons_r)/n])
    chart("Average Confidence",["Quick","Reasoning"],[sum(conf_q)/n,sum(conf_r)/n])

    # CSV export

    csv_buffer=io.StringIO()
    writer=csv.writer(csv_buffer)

    writer.writerow(["claim","truth","quick_verdict","reason_verdict"])

    for i in shared:

        writer.writerow([
            CLAIMS[i]["text"],
            CLAIMS[i]["answer"],
            majority_verdict(quick[i]),
            majority_verdict(reason[i])
        ])

    st.download_button(
        "Download Experiment Data",
        csv_buffer.getvalue(),
        "experiment_results.csv"
    )