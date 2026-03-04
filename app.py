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
    explanation:str
    raw:str

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results={"quick":{}, "reason":{}}

if "judge_results" not in st.session_state:
    st.session_state.judge_results=None

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

    return Result(v,c,"",txt)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def majority(results):
    votes=[r.verdict for r in results]
    return max(set(votes),key=votes.count)

def consistency(results):
    votes=[r.verdict for r in results]
    m=majority(results)
    return votes.count(m)/len(votes)*100

def badge(v):
    if v=="True": return "🟢 TRUE"
    if v=="False": return "🔴 FALSE"
    return "🟡 UNCERTAIN"

def chart(title,values):

    fig,ax=plt.subplots(figsize=(4,3))

    bars=ax.bar(["Quick Mode","Reasoning Mode"],values,color=[COLOR_QUICK,COLOR_REASON])

    for b in bars:
        ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{b.get_height():.1f}",ha='center')

    ax.set_title(title)

    st.pyplot(fig)

# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------

page=st.radio("",["How This Experiment Works","Experiment","Ask Your Own Question","Results"],horizontal=True)

# ---------------------------------------------------------
# INSTRUCTIONS PAGE
# ---------------------------------------------------------

if page=="How This Experiment Works":

    st.header("How the AI Reliability Experiment Works")

    st.write("""
This experiment tests whether **structured reasoning improves AI fact-checking reliability**.
""")

    st.subheader("Two AI Response Modes")

    st.write("""
Quick Mode – AI answers immediately.

Reasoning Mode – AI performs step-by-step reasoning first.
""")

    st.subheader("Hypothesis")

    st.write("""
If AI performs structured reasoning before answering,
it will be:

• More accurate  
• More consistent  
• More likely to say **Uncertain** when information is ambiguous
""")

    st.subheader("Variables")

    st.write("""
Independent Variable  
AI response style

Dependent Variables  
Accuracy  
Consistency  
Confidence  
Uncertainty Rate
""")

    st.subheader("Why Uncertainty Matters")

    st.write("""
Human scientists often say *"I'm not sure"* when evidence is unclear.

AI systems that sometimes answer **Uncertain** may be behaving
more like careful scientific thinkers rather than guessing.
""")

# ---------------------------------------------------------
# EXPERIMENT PAGE
# ---------------------------------------------------------

elif page=="Experiment":

    multi=st.checkbox("Run each claim 3 times (consistency test)",True)
    runs=3 if multi else 1

    if st.button("Run 5 Random Claims (Judge Demo)"):

        progress=st.progress(0)

        results=[]

        for idx,i in enumerate(random.sample(range(20),5)):

            claim=CLAIMS[i]["text"]

            q=[ask_ai(claim,"quick") for _ in range(runs)]
            r=[ask_ai(claim,"reason") for _ in range(runs)]

            st.session_state.results["quick"][i]=q
            st.session_state.results["reason"][i]=r

            results.append((claim,q,r))

            progress.progress((idx+1)/5)

        st.session_state.judge_results=results

    if st.session_state.judge_results:

        for claim,q,r in st.session_state.judge_results:

            st.write("###",claim)

            c1,c2=st.columns(2)

            with c1:

                st.subheader("Quick Mode")

                st.write("Verdict:",badge(majority(q)))
                st.write("Confidence:",round(statistics.mean([x.confidence for x in q]),1),"%")

                st.markdown(q[0].raw)

            with c2:

                st.subheader("Reasoning Mode")

                st.write("Verdict:",badge(majority(r)))
                st.write("Confidence:",round(statistics.mean([x.confidence for x in r]),1),"%")

                st.markdown(r[0].raw)

            st.divider()

    idx=st.session_state.claim_index

    claim=CLAIMS[idx]["text"]

    st.subheader(f"Manual Claim {idx+1}/20")

    st.progress(idx/20)

    st.write(claim)

    col1,col2=st.columns(2)

    if col1.button("Run Quick Mode"):
        st.session_state.results["quick"][idx]=[ask_ai(claim,"quick") for _ in range(runs)]

    if col2.button("Run Reasoning Mode"):
        st.session_state.results["reason"][idx]=[ask_ai(claim,"reason") for _ in range(runs)]

    q=st.session_state.results["quick"].get(idx)
    r=st.session_state.results["reason"].get(idx)

    if q or r:

        c1,c2=st.columns(2)

        if q:
            with c1:
                st.subheader("Quick Mode")
                st.write("Verdict:",badge(majority(q)))
                st.write("Confidence:",round(statistics.mean([x.confidence for x in q]),1),"%")
                st.markdown(q[0].raw)

        if r:
            with c2:
                st.subheader("Reasoning Mode")
                st.write("Verdict:",badge(majority(r)))
                st.write("Confidence:",round(statistics.mean([x.confidence for x in r]),1),"%")
                st.markdown(r[0].raw)

    if q and r:
        if st.button("Next Claim"):
            st.session_state.claim_index+=1

# ---------------------------------------------------------
# ASK PAGE
# ---------------------------------------------------------

elif page=="Ask Your Own Question":

    claim=st.text_input("Enter a science claim")

    if st.button("Analyze"):

        q=ask_ai(claim,"quick")
        r=ask_ai(claim,"reason")

        c1,c2=st.columns(2)

        with c1:
            st.subheader("Quick Mode")
            st.write("Verdict:",badge(q.verdict))
            st.write("Confidence:",q.confidence,"%")
            st.markdown(q.raw)

        with c2:
            st.subheader("Reasoning Mode")
            st.write("Verdict:",badge(r.verdict))
            st.write("Confidence:",r.confidence,"%")
            st.markdown(r.raw)

# ---------------------------------------------------------
# RESULTS PAGE
# ---------------------------------------------------------

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
    unc_q=0
    unc_r=0

    for i in shared:

        truth=CLAIMS[i]["answer"]

        q=quick[i]
        r=reason[i]

        if majority(q)==truth: acc_q+=1
        if majority(r)==truth: acc_r+=1

        cons_q.append(consistency(q))
        cons_r.append(consistency(r))

        if majority(q)=="Uncertain": unc_q+=1
        if majority(r)=="Uncertain": unc_r+=1

    n=len(shared)

    acc_q=acc_q/n*100
    acc_r=acc_r/n*100

    unc_q=unc_q/n*100
    unc_r=unc_r/n*100

    avg_conf_q=statistics.mean([
        statistics.mean([x.confidence for x in quick[i]])
        for i in shared
    ])

    avg_conf_r=statistics.mean([
        statistics.mean([x.confidence for x in reason[i]])
        for i in shared
    ])

    st.header("Results Dashboard")

    st.subheader("Experiment Summary")

    st.write(f"""
Claims tested: **{n}**

Quick Mode Accuracy: **{round(acc_q,1)}%**

Reasoning Mode Accuracy: **{round(acc_r,1)}%**

Accuracy improvement from reasoning: **{round(acc_r-acc_q,1)} percentage points**

Reasoning Uncertainty Rate: **{round(unc_r,1)}%**

Quick Mode Uncertainty Rate: **{round(unc_q,1)}%**
""")

    col1,col2=st.columns(2)

    with col1:
        chart("Accuracy",[acc_q,acc_r])

    with col2:
        chart("Consistency",[statistics.mean(cons_q),statistics.mean(cons_r)])

    col3,col4=st.columns(2)

    with col3:
        chart("Average Confidence",[avg_conf_q,avg_conf_r])

    with col4:
        chart("Uncertainty Rate",[unc_q,unc_r])

    report=f"""
AI Reliability Experiment Report

Claims Tested: {n}

Quick Mode Accuracy: {acc_q:.1f}%
Reasoning Mode Accuracy: {acc_r:.1f}%

Reasoning improved accuracy by {acc_r-acc_q:.1f} percentage points.

Reasoning Uncertainty Rate: {unc_r:.1f}%
Quick Mode Uncertainty Rate: {unc_q:.1f}%
"""

    st.download_button(
        "Download Printable Experiment Report",
        report,
        "experiment_report.txt"
    )