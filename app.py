
import os
import re
import io
import csv
import random
import statistics
from dataclasses import dataclass

import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

st.set_page_config(page_title="AI Reliability Experiment by Jack Eckel", page_icon="🧪", layout="wide")

st.title("AI Reliability Experiment by Jack Eckel")
st.caption("Testing whether structured reasoning improves AI fact-checking reliability by Jack Eckel")

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_TOKENS = 700

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------------------------------
# DATASET
# -----------------------------------------------------

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

# -----------------------------------------------------
# PROMPTS
# -----------------------------------------------------

SYSTEM = "You are a careful science fact checker."

QUICK_PROMPT = """
VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

REASON_PROMPT = """
REASONING:
Step 1: Identify scientific principle
Step 2: Compare with established science

VERDICT: <True|False|Uncertain>
CONFIDENCE: <0-100>
EXPLANATION: <short explanation>

Claim: "{claim}"
"""

# -----------------------------------------------------
# DATA CLASS
# -----------------------------------------------------

@dataclass
class Result:
    verdict:str
    confidence:int
    explanation:str
    raw:str

# -----------------------------------------------------
# SESSION STATE
# -----------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results={"quick":{}, "reason":{}}

if "judge_results" not in st.session_state:
    st.session_state.judge_results=None

if "claim_index" not in st.session_state:
    st.session_state.claim_index=0

# -----------------------------------------------------
# HELPERS
# -----------------------------------------------------

def parse(txt):

    v=re.search(r"VERDICT:\s*(True|False|Uncertain)",txt)
    c=re.search(r"CONFIDENCE:\s*(\d+)",txt)
    e=re.search(r"EXPLANATION:\s*(.*)",txt)

    verdict=v.group(1) if v else "Uncertain"
    confidence=int(c.group(1)) if c else 50
    explanation=e.group(1) if e else ""

    return verdict,confidence,explanation

def ask_ai(claim,mode):

    prompt=QUICK_PROMPT if mode=="quick" else REASON_PROMPT

    with st.spinner("Analyzing with AI..."):

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

def badge(v):

    if v=="True": return "🟢 TRUE"
    if v=="False": return "🔴 FALSE"
    return "🟡 UNCERTAIN"

def majority(results):

    votes=[r.verdict for r in results]
    return max(set(votes),key=votes.count)

def consistency(results):

    votes=[r.verdict for r in results]
    m=majority(results)
    return votes.count(m)/len(votes)*100

# -----------------------------------------------------
# NAVIGATION
# -----------------------------------------------------

page=st.radio("",["How This Experiment Works","Experiment","Ask Your Own Question","Results"],horizontal=True)

# -----------------------------------------------------
# INSTRUCTIONS
# -----------------------------------------------------

if page=="How This Experiment Works":

    st.header("AI Reliability Experiment By Jack Eckel - How This Experiment Works")

    st.write("""
This experiment tests whether **structured reasoning improves AI fact-checking reliability**.

Two response modes are compared:

**Quick Mode**  
The AI answers immediately.

**Reasoning Mode**  
The AI performs step-by-step reasoning first.
""")

    st.write("""
Metrics measured:

• Accuracy  
• Confidence  
• Consistency across runs
""")

    st.info("This experiment demonstrates that reasoning-based AI systems may produce more reliable results than immediate responses.")

# -----------------------------------------------------
# EXPERIMENT PAGE
# -----------------------------------------------------

elif page=="Experiment":

    st.header("Experiment")

    multi=st.checkbox("Run each claim 3 times (consistency test)",True)
    runs=3 if multi else 1

    col1,col2=st.columns(2)

    if col1.button("Run 5 Random Claims (Judge Demo)"):

        batch=[]

        for i in random.sample(range(20),5):

            claim=CLAIMS[i]["text"]
            truth=CLAIMS[i]["answer"]

            q=[ask_ai(claim,"quick") for _ in range(runs)]
            r=[ask_ai(claim,"reason") for _ in range(runs)]

            st.session_state.results["quick"][i]=q
            st.session_state.results["reason"][i]=r

            batch.append((claim,truth,q,r))

        st.session_state.judge_results=batch

    if col2.button("Restart Experiment"):

        st.session_state.results={"quick":{}, "reason":{}}
        st.session_state.judge_results=None
        st.session_state.claim_index=0

    # judge display

    if st.session_state.judge_results:

        st.subheader("Judge Demo Results")

        for claim,truth,q,r in st.session_state.judge_results:

            st.write("###",claim)

            c1,c2=st.columns(2)

            with c1:
                st.write("Quick Mode")
                st.write(badge(majority(q)))
                st.write("Confidence:",round(statistics.mean([x.confidence for x in q]),1))
                st.write("Consistency:",round(consistency(q),1),"%")
                st.write(q[0].explanation)

            with c2:
                st.write("Reasoning Mode")
                st.write(badge(majority(r)))
                st.write("Confidence:",round(statistics.mean([x.confidence for x in r]),1))
                st.write("Consistency:",round(consistency(r),1),"%")
                st.write(r[0].explanation)

            st.divider()

    # manual experiment

    idx=st.session_state.claim_index

    claim=CLAIMS[idx]["text"]

    st.subheader(f"Manual Experiment Claim {idx+1}/20")

    progress=(idx)/20
    st.progress(progress)

    st.write(claim)

    col1,col2=st.columns(2)

    if col1.button("Run Quick Mode"):

        st.session_state.results["quick"][idx]=[ask_ai(claim,"quick") for _ in range(runs)]

    if col2.button("Run Reasoning Mode"):

        st.session_state.results["reason"][idx]=[ask_ai(claim,"reason") for _ in range(runs)]

    q=st.session_state.results["quick"].get(idx)
    r=st.session_state.results["reason"].get(idx)

    if q and r:

        st.write("### Results")

        c1,c2=st.columns(2)

        with c1:
            st.write("Quick Mode",badge(majority(q)))

        with c2:
            st.write("Reasoning Mode",badge(majority(r)))

        if st.button("Next Claim"):

            if idx<19:
                st.session_state.claim_index+=1

# -----------------------------------------------------
# ASK QUESTION
# -----------------------------------------------------

elif page=="Ask Your Own Question":

    claim=st.text_input("Enter a science claim")

    if st.button("Analyze"):

        q=ask_ai(claim,"quick")
        r=ask_ai(claim,"reason")

        c1,c2=st.columns(2)

        with c1:
            st.write("Quick Mode",badge(q.verdict),q.confidence)
            st.write(q.explanation)

        with c2:
            st.write("Reasoning Mode",badge(r.verdict),r.confidence)
            st.write(r.explanation)

# -----------------------------------------------------
# RESULTS DASHBOARD
# -----------------------------------------------------

else:

    quick=st.session_state.results["quick"]
    reason=st.session_state.results["reason"]

    shared=set(quick)&set(reason)

    if not shared:

        st.warning("Run experiment first.")
        st.stop()

    acc_q=0
    acc_r=0

    cons_q=[]
    cons_r=[]

    for i in shared:

        truth=CLAIMS[i]["answer"]

        q=quick[i]
        r=reason[i]

        if majority(q)==truth: acc_q+=1
        if majority(r)==truth: acc_r+=1

        cons_q.append(consistency(q))
        cons_r.append(consistency(r))

    n=len(shared)

    acc_q=acc_q/n*100
    acc_r=acc_r/n*100

    st.header("Results Dashboard")

    st.success(f"Reasoning Mode improved accuracy by {round(acc_r-acc_q,1)}%")

    def chart(title,values):

        fig,ax=plt.subplots(figsize=(4,3))

        bars=ax.bar(["Quick","Reasoning"],values)

        for b in bars:
            ax.text(b.get_x()+b.get_width()/2,b.get_height(),round(b.get_height(),1),ha='center')

        ax.set_title(title)

        st.pyplot(fig)

    col1,col2=st.columns(2)

    with col1:
        chart("Accuracy",[acc_q,acc_r])

    with col2:
        chart("Consistency",[statistics.mean(cons_q),statistics.mean(cons_r)])