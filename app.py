import os
import re
import random
import statistics
from dataclasses import dataclass

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

plt.style.use("seaborn-v0_8-whitegrid")

COLOR_QUICK="#1f77b4"
COLOR_REASON="#ff7f0e"

# ---------------------------------------------------------
# CLAIM DATASET
# ---------------------------------------------------------

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
{"text":"Glass is a solid.","answer":"Uncertain"}

]

# ---------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------

SYSTEM="You are a careful science fact checker."

QUICK_PROMPT="""
VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

REASON_PROMPT="""
Think step-by-step and explain reasoning using bullet points.

REASONING:
• Identify the scientific principle
• Apply the principle to the claim
• Determine if the claim conflicts with known science

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

# ---------------------------------------------------------
# DATA CLASS
# ---------------------------------------------------------

@dataclass
class Result:
    verdict:str
    confidence:int
    raw:str

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results={"quick":{},"reason":{}}

if "claim_index" not in st.session_state:
    st.session_state.claim_index=0

# ---------------------------------------------------------
# AI CALL
# ---------------------------------------------------------

def parse(txt):

    v=re.search(r"VERDICT:\s*(True|False|Uncertain)",txt)
    c=re.search(r"CONFIDENCE:\s*(\d+)",txt)

    verdict=v.group(1) if v else "Uncertain"
    confidence=int(c.group(1)) if c else 50

    return verdict,confidence

def ask_ai(claim,mode):

    prompt=QUICK_PROMPT if mode=="quick" else REASON_PROMPT

    with st.spinner("Analyzing claim with AI..."):

        r=client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":SYSTEM},
                {"role":"user","content":prompt.format(claim=claim)}
            ]
        )

    txt=r.choices[0].message.content
    v,c=parse(txt)

    return Result(v,c,txt)

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def majority(results):
    votes=[r.verdict for r in results]
    return max(set(votes),key=votes.count)

def consistency(results):
    votes=[r.verdict for r in results]
    m=majority(results)
    return votes.count(m)/len(votes)*100

def responsible_score(pred,truth):

    if pred==truth:
        return 1
    if pred=="Uncertain":
        return 0.5
    return 0

# ---------------------------------------------------------
# CHART
# ---------------------------------------------------------

def chart(title,values):

    fig,ax=plt.subplots(figsize=(4,3))

    bars=ax.bar(["Quick","Reasoning"],values,color=[COLOR_QUICK,COLOR_REASON])

    for b in bars:
        ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{b.get_height():.1f}",ha='center')

    ax.set_title(title)
    st.pyplot(fig)

# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------

page=st.radio("",["Instructions","Experiment","Ask Your Own Question","Results"],horizontal=True)

# ---------------------------------------------------------
# INSTRUCTIONS
# ---------------------------------------------------------

if page=="Instructions":

    st.header("How to Use the Experiment")

    st.write("""
This project tests whether **structured reasoning improves AI fact-checking reliability**.

Two AI modes are compared:

Quick Mode — AI answers immediately.

Reasoning Mode — AI uses step-by-step reasoning first.
""")

    st.subheader("How to Run the Experiment")

    st.write("""
1. Click **Run 5 Random Claims** to demonstrate the system quickly.

2. Use the **Manual Experiment** to test all 20 scientific claims.

3. Compare Quick Mode and Reasoning Mode answers.

4. Open the **Results Dashboard** to see the experiment analysis.
""")

    st.subheader("What the Charts Mean")

    st.write("""
Accuracy — percent of correct answers.

Consistency — how often repeated runs give the same answer.

Average Confidence — how confident the AI is.

Uncertainty Rate — how often the AI admits uncertainty.

Responsible Answer Score — rewards appropriate uncertainty.

Overconfidence Rate — measures when AI should say uncertain but doesn't.
""")

    st.info("This experiment investigates whether reasoning makes AI behave more like careful human scientists.")

# ---------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------

elif page=="Experiment":

    if st.button("Reset Experiment"):
        st.session_state.results={"quick":{},"reason":{}}
        st.session_state.claim_index=0

    runs=3 if st.checkbox("Run each claim 3 times") else 1

    if st.button("Run 5 Random Claims"):

        progress=st.progress(0)

        for idx,i in enumerate(random.sample(range(20),5)):

            claim=CLAIMS[i]["text"]

            st.session_state.results["quick"][i]=[ask_ai(claim,"quick") for _ in range(runs)]
            st.session_state.results["reason"][i]=[ask_ai(claim,"reason") for _ in range(runs)]

            progress.progress((idx+1)/5)

    idx=st.session_state.claim_index
    claim=CLAIMS[idx]["text"]

    st.subheader(f"Manual Claim {idx+1}/20")
    st.progress(idx/20)
    st.write(claim)

    c1,c2=st.columns(2)

    if c1.button("Run Quick"):
        st.session_state.results["quick"][idx]=[ask_ai(claim,"quick") for _ in range(runs)]

    if c2.button("Run Reasoning"):
        st.session_state.results["reason"][idx]=[ask_ai(claim,"reason") for _ in range(runs)]

    q=st.session_state.results["quick"].get(idx)
    r=st.session_state.results["reason"].get(idx)

    if q or r:

        col1,col2=st.columns(2)

        if q:
            with col1:
                st.subheader("Quick Mode")
                st.write(majority(q),statistics.mean([x.confidence for x in q]),"%")
                st.markdown(q[0].raw)

        if r:
            with col2:
                st.subheader("Reasoning Mode")
                st.write(majority(r),statistics.mean([x.confidence for x in r]),"%")
                st.markdown(r[0].raw)

    if q and r:
        if st.button("Next Claim"):
            st.session_state.claim_index+=1

# ---------------------------------------------------------
# ASK PAGE
# ---------------------------------------------------------

elif page=="Ask Your Own Question":

    claim=st.text_input("Enter a scientific claim")

    if st.button("Analyze Claim"):

        q=ask_ai(claim,"quick")
        r=ask_ai(claim,"reason")

        c1,c2=st.columns(2)

        with c1:
            st.subheader("Quick Mode")
            st.write(q.verdict,q.confidence,"%")
            st.markdown(q.raw)

        with c2:
            st.subheader("Reasoning Mode")
            st.write(r.verdict,r.confidence,"%")
            st.markdown(r.raw)

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

else:

    quick=st.session_state.results["quick"]
    reason=st.session_state.results["reason"]

    shared=set(quick)&set(reason)

    if not shared:
        st.warning("Run the experiment first.")
        st.stop()

    acc_q=acc_r=unc_q=unc_r=ras_q=ras_r=over_q=over_r=0
    cons_q=[]
    cons_r=[]

    for i in shared:

        truth=CLAIMS[i]["answer"]

        pred_q=majority(quick[i])
        pred_r=majority(reason[i])

        if pred_q==truth: acc_q+=1
        if pred_r==truth: acc_r+=1

        ras_q+=responsible_score(pred_q,truth)
        ras_r+=responsible_score(pred_r,truth)

        if pred_q=="Uncertain": unc_q+=1
        if pred_r=="Uncertain": unc_r+=1

        if truth=="Uncertain":

            if pred_q!="Uncertain": over_q+=1
            if pred_r!="Uncertain": over_r+=1

        cons_q.append(consistency(quick[i]))
        cons_r.append(consistency(reason[i]))

    n=len(shared)

    acc_q=acc_q/n*100
    acc_r=acc_r/n*100
    ras_q=ras_q/n*100
    ras_r=ras_r/n*100
    unc_q=unc_q/n*100
    unc_r=unc_r/n*100
    over_q=over_q/n*100
    over_r=over_r/n*100

    st.header("Results Dashboard")

    st.write("This dashboard compares AI performance between Quick Mode and Reasoning Mode.")

    c1,c2,c3=st.columns(3)

    with c1: chart("Accuracy",[acc_q,acc_r])
    with c2: chart("Consistency",[statistics.mean(cons_q),statistics.mean(cons_r)])
    with c3: chart("Uncertainty Rate",[unc_q,unc_r])

    c4,c5,c6=st.columns(3)

    with c4: chart("Responsible Score",[ras_q,ras_r])
    with c5: chart("Overconfidence Rate",[over_q,over_r])
    with c6: chart("Average Confidence",[
        statistics.mean([statistics.mean([x.confidence for x in quick[i]]) for i in shared]),
        statistics.mean([statistics.mean([x.confidence for x in reason[i]]) for i in shared])
    ])

    st.subheader("Interpretation")

    st.write("""
If Reasoning Mode shows higher Responsible Answer Score,
higher consistency, and lower overconfidence,
it suggests reasoning makes AI behave more like a careful scientist.
""")