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
{"text":"Glass is a solid.","answer":"Uncertain"},
]


# ---------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------

SYSTEM = "You are a careful science fact checker."

QUICK_PROMPT = """
VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

REASON_PROMPT = """
Think step-by-step.

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
REASONING:
• step
• step
• step
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
    st.session_state.judge_last_run = []

if "claim_index" not in st.session_state:
    st.session_state.claim_index = 0


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def badge(v):
    if v == "True":
        return "🟢 TRUE"
    if v == "False":
        return "🔴 FALSE"
    return "🟡 UNCERTAIN"


def parse(txt):
    v = re.search(r"VERDICT:\s*(True|False|Uncertain)", txt)
    c = re.search(r"CONFIDENCE:\s*(\d+)", txt)

    verdict = v.group(1) if v else "Uncertain"
    confidence = int(c.group(1)) if c else 50

    return verdict, confidence


def ask_ai(claim, mode):

    prompt = QUICK_PROMPT if mode == "quick" else REASON_PROMPT

    with st.spinner("Analyzing claim with AI..."):

        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt.format(claim=claim)}
            ]
        )

    txt = r.choices[0].message.content

    v, c = parse(txt)

    return Result(v, c, txt)


def majority(results):

    votes = [r.verdict for r in results]

    return max(set(votes), key=votes.count)


def consistency(results):

    votes = [r.verdict for r in results]

    m = majority(results)

    return votes.count(m) / len(votes) * 100


def mean_conf(results):

    return statistics.mean([x.confidence for x in results]) if results else 0


def responsible_score(pred, truth):

    if pred == truth:
        return 1

    if pred == "Uncertain":
        return .5

    return 0


def small_bar(title, values):

    fig, ax = plt.subplots(figsize=(4,3))

    bars = ax.bar(
        ["Quick","Reasoning"],
        values,
        color=[COLOR_QUICK,COLOR_REASON]
    )

    for b in bars:
        ax.text(
            b.get_x()+b.get_width()/2,
            b.get_height(),
            f"{b.get_height():.1f}",
            ha="center"
        )

    ax.set_title(title)

    st.pyplot(fig)


def calibration_chart(title, accuracy, confidence):

    fig, ax = plt.subplots(figsize=(4,3))

    width = .35

    x=[0,1]

    ax.bar([i-width/2 for i in x],accuracy,width,label="Accuracy")
    ax.bar([i+width/2 for i in x],confidence,width,label="Confidence")

    ax.set_xticks(x)
    ax.set_xticklabels(["Quick","Reasoning"])

    ax.legend()

    ax.set_title(title)

    st.pyplot(fig)


# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------

page = st.radio("",["Instructions","Experiment","Ask Your Own Question","Results"],horizontal=True)


# ---------------------------------------------------------
# INSTRUCTIONS
# ---------------------------------------------------------

if page=="Instructions":

    st.header("How To Run The Experiment")

    st.markdown("""
**Goal**

Test whether reasoning improves AI reliability.

**Judge Demo**

1. Go to Experiment  
2. Click Run 5 Random Claims  
3. Compare Quick vs Reasoning  
4. Show Results dashboard
""")

    st.info("Key idea: In science the most responsible answer can sometimes be **Uncertain**.")


# ---------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------

elif page=="Experiment":

    if st.button("Reset Experiment"):
        st.session_state.results={"quick":{},"reason":{}}
        st.session_state.judge_last_run=[]
        st.session_state.claim_index=0


    run_three=st.checkbox("Run each claim 3 times",True)

    runs=3 if run_three else 1

    st.subheader("Judge Demo Mode")

    if st.button("Run 5 Random Claims (Judge Demo)"):

        idxs=random.sample(range(len(CLAIMS)),5)

        st.session_state.judge_last_run=idxs

        for i in idxs:

            claim=CLAIMS[i]["text"]

            st.session_state.results["quick"][i]=[ask_ai(claim,"quick") for _ in range(runs)]

            st.session_state.results["reason"][i]=[ask_ai(claim,"reason") for _ in range(runs)]


    if st.session_state.judge_last_run:

        for i in st.session_state.judge_last_run:

            claim=CLAIMS[i]["text"]

            q=st.session_state.results["quick"][i]

            r=st.session_state.results["reason"][i]

            st.markdown(f"### {claim}")

            c1,c2=st.columns(2)

            with c1:

                st.subheader("Quick")

                st.write("Verdict:",badge(majority(q)))

                st.write("Confidence:",f"{mean_conf(q):.1f}%")

                st.markdown(q[0].raw)

            with c2:

                st.subheader("Reasoning")

                st.write("Verdict:",badge(majority(r)))

                st.write("Confidence:",f"{mean_conf(r):.1f}%")

                st.markdown(r[0].raw)


    st.subheader("Manual Experiment")

    idx=st.session_state.claim_index

    claim=CLAIMS[idx]["text"]

    st.write(claim)

    if st.button("Quick"):

        st.session_state.results["quick"][idx]=[ask_ai(claim,"quick") for _ in range(runs)]

    if st.button("Reasoning"):

        st.session_state.results["reason"][idx]=[ask_ai(claim,"reason") for _ in range(runs)]

    if st.button("Next"):

        st.session_state.claim_index+=1


# ---------------------------------------------------------
# ASK YOUR OWN
# ---------------------------------------------------------

elif page=="Ask Your Own Question":

    claim=st.text_input("Enter a scientific claim")

    if st.button("Analyze"):

        q=[ask_ai(claim,"quick")]

        r=[ask_ai(claim,"reason")]

        c1,c2=st.columns(2)

        with c1:

            st.write("Quick")

            st.write(badge(q[0].verdict))

            st.markdown(q[0].raw)

        with c2:

            st.write("Reasoning")

            st.write(badge(r[0].verdict))

            st.markdown(r[0].raw)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

else:

    st.header("Results Dashboard")

    quick=st.session_state.results["quick"]

    reason=st.session_state.results["reason"]

    shared=set(quick)&set(reason)

    if not shared:

        st.warning("Run experiment first")

        st.stop()

    rows=[]

    acc_q=0
    acc_r=0

    unc_q=0
    unc_r=0

    ras_q=0
    ras_r=0

    over_q=0
    over_r=0

    for i in shared:

        truth=CLAIMS[i]["answer"]

        q=quick[i]

        r=reason[i]

        pq=majority(q)

        pr=majority(r)

        if pq==truth: acc_q+=1
        if pr==truth: acc_r+=1

        if pq=="Uncertain": unc_q+=1
        if pr=="Uncertain": unc_r+=1

        ras_q+=responsible_score(pq,truth)
        ras_r+=responsible_score(pr,truth)

        rows.append({
        "claim":CLAIMS[i]["text"],
        "truth":truth,
        "quick":pq,
        "reason":pr
        })


    n=len(shared)

    acc_q=acc_q/n*100
    acc_r=acc_r/n*100

    unc_q=unc_q/n*100
    unc_r=unc_r/n*100

    ras_q=ras_q/n*100
    ras_r=ras_r/n*100


    st.subheader("Charts")

    c1,c2,c3=st.columns(3)

    with c1: small_bar("Accuracy",[acc_q,acc_r])
    with c2: small_bar("Uncertainty Rate",[unc_q,unc_r])
    with c3: small_bar("Responsible Score",[ras_q,ras_r])


    calibration_chart("Confidence Calibration",[acc_q,acc_r],[50,50])


    st.subheader("Data")

    df=pd.DataFrame(rows)

    st.dataframe(df)

    csv=df.to_csv(index=False).encode()

    st.download_button("Download CSV",csv,"results.csv")


    st.subheader("Printable Report")

    report=f"""
AI Reliability Experiment

Claims tested: {n}

Accuracy Quick: {acc_q:.1f}
Accuracy Reasoning: {acc_r:.1f}

Responsible Answer Score Quick: {ras_q:.1f}
Responsible Answer Score Reasoning: {ras_r:.1f}
"""

    st.download_button("Download Report",report,"report.txt")