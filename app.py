import os
import re
import random
import statistics
import pandas as pd
from dataclasses import dataclass

import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

st.set_page_config(page_title="AI Reliability Experiment", page_icon="🧪", layout="wide")

st.title("AI Reliability Experiment by Jack Eckel")
st.caption("Testing whether structured reasoning improves AI fact-checking reliability")

MODEL="gpt-4o-mini"
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

plt.style.use("seaborn-v0_8-whitegrid")

COLOR_QUICK="#1f77b4"
COLOR_REASON="#ff7f0e"

# ---------------------------------------------------------
# DATASET
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

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

# ---------------------------------------------------------
# STRUCTURES
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

def majority(results):
    votes=[r.verdict for r in results]
    return max(set(votes),key=votes.count)

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

    st.header("How This Experiment Works")

    st.write("""
This project tests whether **reasoning improves AI reliability**.

Two modes are compared:

Quick Mode – AI answers immediately.

Reasoning Mode – AI analyzes the claim step-by-step before answering.
""")

    st.subheader("Running the Experiment")

    st.write("""
Judge Mode  
Runs 5 random claims automatically.

Manual Mode  
Run all 20 claims individually.
""")

    st.subheader("Results Dashboard")

    st.write("""
Accuracy – percent correct

Consistency – repeated answer stability

Confidence – model certainty

Uncertainty Rate – when AI admits uncertainty

Responsible Score – rewards correct uncertainty

Overconfidence – when AI should say uncertain but doesn't
""")

    st.info("The goal is to test whether structured reasoning produces more reliable scientific answers.")

# ---------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------

elif page=="Experiment":

    if st.button("Reset Experiment"):
        st.session_state.results={"quick":{},"reason":{}}
        st.session_state.claim_index=0

    runs=3 if st.checkbox("Run each claim 3 times") else 1

    st.header("Judge Demo Mode")

    if st.button("Run 5 Random Claims"):

        for i in random.sample(range(20),5):

            claim=CLAIMS[i]["text"]

            q=[ask_ai(claim,"quick") for _ in range(runs)]
            r=[ask_ai(claim,"reason") for _ in range(runs)]

            st.session_state.results["quick"][i]=q
            st.session_state.results["reason"][i]=r

            st.subheader(claim)

            c1,c2=st.columns(2)

            with c1:
                st.write(badge(majority(q)))
                st.write("Confidence:",statistics.mean([x.confidence for x in q]),"%")
                st.markdown(q[0].raw)

            with c2:
                st.write(badge(majority(r)))
                st.write("Confidence:",statistics.mean([x.confidence for x in r]),"%")
                st.markdown(r[0].raw)

    st.header("Manual Experiment")

    idx=st.session_state.claim_index
    claim=CLAIMS[idx]["text"]

    st.write(claim)

    col1,col2=st.columns(2)

    if col1.button("Run Quick"):
        st.session_state.results["quick"][idx]=[ask_ai(claim,"quick")]

    if col2.button("Run Reasoning"):
        st.session_state.results["reason"][idx]=[ask_ai(claim,"reason")]

    q=st.session_state.results["quick"].get(idx)
    r=st.session_state.results["reason"].get(idx)

    if q or r:

        c1,c2=st.columns(2)

        if q:
            with c1:
                st.write(badge(q[0].verdict))
                st.write("Confidence:",q[0].confidence,"%")
                st.markdown(q[0].raw)

        if r:
            with c2:
                st.write(badge(r[0].verdict))
                st.write("Confidence:",r[0].confidence,"%")
                st.markdown(r[0].raw)

    if q and r:
        if st.button("Next Claim"):
            st.session_state.claim_index+=1

# ---------------------------------------------------------
# ASK PAGE
# ---------------------------------------------------------

elif page=="Ask Your Own Question":

    claim=st.text_input("Enter a scientific claim")

    if st.button("Analyze"):

        q=ask_ai(claim,"quick")
        r=ask_ai(claim,"reason")

        c1,c2=st.columns(2)

        with c1:
            st.write(badge(q.verdict))
            st.write("Confidence:",q.confidence,"%")
            st.markdown(q.raw)

        with c2:
            st.write(badge(r.verdict))
            st.write("Confidence:",r.confidence,"%")
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

    data=[]

    for i in shared:

        truth=CLAIMS[i]["answer"]

        q=majority(quick[i])
        r=majority(reason[i])

        data.append({
            "claim":CLAIMS[i]["text"],
            "truth":truth,
            "quick":q,
            "reason":r
        })

    df=pd.DataFrame(data)

    st.header("Results Dashboard")

    st.dataframe(df)

    csv=df.to_csv(index=False)

    st.download_button(
        "Download Results CSV",
        csv,
        "ai_experiment_results.csv",
        "text/csv"
    )