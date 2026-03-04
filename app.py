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

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
REASONING:
• bullet
• bullet
• bullet
EXPLANATION: short explanation

Claim: "{claim}"
"""

# ---------------------------------------------------------
# DATA STRUCTURE
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

if "judge_last_run" not in st.session_state:
    st.session_state.judge_last_run=[]

if "claim_index" not in st.session_state:
    st.session_state.claim_index=0

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def badge(v):
    if v=="True":
        return "🟢 TRUE"
    if v=="False":
        return "🔴 FALSE"
    return "🟡 UNCERTAIN"

def parse(txt):
    v=re.search(r"VERDICT:\s*(True|False|Uncertain)",txt)
    c=re.search(r"CONFIDENCE:\s*(\d+)",txt)
    verdict=v.group(1) if v else "Uncertain"
    conf=int(c.group(1)) if c else 50
    return verdict,conf

def ask_ai(claim,mode):
    prompt=QUICK_PROMPT if mode=="quick" else REASON_PROMPT
    with st.spinner("Analyzing claim..."):
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

def majority(results):
    votes=[r.verdict for r in results]
    return max(set(votes),key=votes.count)

def consistency(results):
    votes=[r.verdict for r in results]
    m=majority(results)
    return votes.count(m)/len(votes)*100

def mean_conf(results):
    return statistics.mean([x.confidence for x in results])

# ---------------------------------------------------------
# CHART HELPERS
# ---------------------------------------------------------
def small_bar(title,values):

    fig,ax=plt.subplots(figsize=(4.2,2.9))

    bars=ax.bar(["Quick","Reasoning"],values,
    color=[COLOR_QUICK,COLOR_REASON])

    for b in bars:
        ax.text(b.get_x()+b.get_width()/2,b.get_height(),
        f"{b.get_height():.1f}",ha="center")

    ax.set_ylim(0,max(100,max(values)*1.2))
    ax.set_title(title)

    st.pyplot(fig,clear_figure=True)

def calibration_chart(acc,conf):

    fig,ax=plt.subplots(figsize=(4.2,2.9))

    width=0.35

    bars1=ax.bar([-width/2,1-width/2],[acc[0],acc[1]],
    width,label="Accuracy",color="#2ca02c")

    bars2=ax.bar([width/2,1+width/2],[conf[0],conf[1]],
    width,label="Confidence",color="#d62728")

    ax.set_xticks([0,1])
    ax.set_xticklabels(["Quick","Reasoning"])
    ax.set_ylim(0,100)
    ax.set_title("Confidence Calibration")
    ax.legend()

    st.pyplot(fig,clear_figure=True)

# ---------------------------------------------------------
# AI ANALYSIS
# ---------------------------------------------------------
def generate_ai_summary(metrics):

    prompt=f"""
Analyze the results of a science experiment comparing two AI modes.

Accuracy Quick: {metrics['acc_q']}
Accuracy Reasoning: {metrics['acc_r']}

Consistency Quick: {metrics['cons_q']}
Consistency Reasoning: {metrics['cons_r']}

Confidence Quick: {metrics['conf_q']}
Confidence Reasoning: {metrics['conf_r']}

Uncertainty Quick: {metrics['unc_q']}
Uncertainty Reasoning: {metrics['unc_r']}

Responsible Score Quick: {metrics['ras_q']}
Responsible Score Reasoning: {metrics['ras_r']}

Overconfidence Quick: {metrics['over_q']}
Overconfidence Reasoning: {metrics['over_r']}

Write a short scientific interpretation explaining the results.
"""

    r=client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system","content":"You analyze scientific experiments."},
            {"role":"user","content":prompt}
        ]
    )

    return r.choices[0].message.content

# ---------------------------------------------------------
# NAV
# ---------------------------------------------------------
page=st.radio("",["Instructions","Experiment","Ask Your Own Question","Results"],horizontal=True)

# ---------------------------------------------------------
# INSTRUCTIONS
# ---------------------------------------------------------
if page=="Instructions":

    st.header("How This Experiment Works")

    st.write("""
This experiment compares two AI behaviors:

Quick Mode — answers immediately

Reasoning Mode — analyzes step-by-step
""")

    st.write("""
Metrics measured:

Accuracy  
Consistency  
Confidence  
Uncertainty Rate  
Responsible Answer Score  
Overconfidence Rate
""")

# ---------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------
elif page=="Experiment":

    runs=3 if st.checkbox("Run 3 times (consistency test)",value=True) else 1

    st.subheader("Judge Demo")

    if st.button("Run 5 Random Claims"):

        idxs=random.sample(range(len(CLAIMS)),5)
        st.session_state.judge_last_run=idxs

        prog=st.progress(0)

        for k,i in enumerate(idxs,start=1):

            claim=CLAIMS[i]["text"]

            st.session_state.results["quick"][i]=[
                ask_ai(claim,"quick") for _ in range(runs)
            ]

            st.session_state.results["reason"][i]=[
                ask_ai(claim,"reason") for _ in range(runs)
            ]

            prog.progress(k/5)

    for i in st.session_state.judge_last_run:

        claim=CLAIMS[i]["text"]
        q=st.session_state.results["quick"][i]
        r=st.session_state.results["reason"][i]

        st.markdown(f"### {claim}")

        c1,c2=st.columns(2)

        with c1:
            st.write("Verdict:",badge(majority(q)))
            st.write("Confidence:",f"{mean_conf(q):.1f}%")
            st.markdown(q[0].raw)

        with c2:
            st.write("Verdict:",badge(majority(r)))
            st.write("Confidence:",f"{mean_conf(r):.1f}%")
            st.markdown(r[0].raw)

# ---------------------------------------------------------
# ASK
# ---------------------------------------------------------
elif page=="Ask Your Own Question":

    claim=st.text_input("Enter a scientific claim")

    if st.button("Analyze") and claim:

        q=[ask_ai(claim,"quick")]
        r=[ask_ai(claim,"reason")]

        c1,c2=st.columns(2)

        with c1:
            st.write(badge(q[0].verdict))
            st.write(q[0].raw)

        with c2:
            st.write(badge(r[0].verdict))
            st.write(r[0].raw)

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------
else:

    st.header("Results Dashboard")

    quick=st.session_state.results["quick"]
    reason=st.session_state.results["reason"]

    shared=set(quick)&set(reason)

    if not shared:
        st.warning("Run the experiment first.")
        st.stop()

    acc_q=acc_r=0
    unc_q=unc_r=0
    ras_q=ras_r=0
    over_q=over_r=0

    cons_q=[]
    cons_r=[]

    conf_q=[]
    conf_r=[]

    for i in shared:

        truth=CLAIMS[i]["answer"]

        q=quick[i]
        r=reason[i]

        pred_q=majority(q)
        pred_r=majority(r)

        if pred_q==truth: acc_q+=1
        if pred_r==truth: acc_r+=1

        if pred_q=="Uncertain": unc_q+=1
        if pred_r=="Uncertain": unc_r+=1

        ras_q+=1 if pred_q==truth else 0
        ras_r+=1 if pred_r==truth else 0

        if truth=="Uncertain":
            if pred_q!="Uncertain": over_q+=1
            if pred_r!="Uncertain": over_r+=1

        cons_q.append(consistency(q))
        cons_r.append(consistency(r))

        conf_q.append(mean_conf(q))
        conf_r.append(mean_conf(r))

    n=len(shared)

    acc_q=acc_q/n*100
    acc_r=acc_r/n*100

    unc_q=unc_q/n*100
    unc_r=unc_r/n*100

    ras_q=ras_q/n*100
    ras_r=ras_r/n*100

    over_q=over_q/n*100
    over_r=over_r/n*100

    avg_conf_q=statistics.mean(conf_q)
    avg_conf_r=statistics.mean(conf_r)

    st.subheader("Charts")

    c1,c2,c3=st.columns(3)

    with c1: small_bar("Accuracy",[acc_q,acc_r])
    with c2: small_bar("Consistency",[statistics.mean(cons_q),statistics.mean(cons_r)])
    with c3: small_bar("Confidence",[avg_conf_q,avg_conf_r])

    c4,c5,c6=st.columns(3)

    with c4: small_bar("Uncertainty Rate",[unc_q,unc_r])
    with c5: small_bar("Responsible Score",[ras_q,ras_r])
    with c6: small_bar("Overconfidence",[over_q,over_r])

    st.subheader("Confidence Calibration")

    calibration_chart(
        [acc_q,acc_r],
        [avg_conf_q,avg_conf_r]
    )

    metrics={
        "acc_q":acc_q,
        "acc_r":acc_r,
        "cons_q":statistics.mean(cons_q),
        "cons_r":statistics.mean(cons_r),
        "conf_q":avg_conf_q,
        "conf_r":avg_conf_r,
        "unc_q":unc_q,
        "unc_r":unc_r,
        "ras_q":ras_q,
        "ras_r":ras_r,
        "over_q":over_q,
        "over_r":over_r
    }

    st.subheader("AI Analysis of the Results")

    col1,col2=st.columns([3,1])

    with col2:
        if st.button("Regenerate AI Analysis"):
            st.session_state.pop("ai_summary",None)

    if "ai_summary" not in st.session_state:
        with st.spinner("Generating AI analysis..."):
            st.session_state.ai_summary=generate_ai_summary(metrics)

    st.write(st.session_state.ai_summary)

    report=f"""
AI Reliability Experiment Report

Accuracy Quick: {acc_q}
Accuracy Reasoning: {acc_r}

Responsible Score Quick: {ras_q}
Responsible Score Reasoning: {ras_r}
"""

    st.download_button(
        "Download Printable Report",
        report,
        file_name="ai_experiment_report.txt"
    )