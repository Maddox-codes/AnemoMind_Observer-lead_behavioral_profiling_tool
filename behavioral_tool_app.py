
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from worklife_rules import get_work_life_compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    anxiety_model = joblib.load(os.path.join(BASE_DIR, 'anxiety_model.pkl'))
    speech_speed_classes = joblib.load(os.path.join(BASE_DIR, 'speech_speed_classes.pkl'))
    stability_model = joblib.load(os.path.join(BASE_DIR, 'stability_model.pkl'))
    integrity_model = joblib.load(os.path.join(BASE_DIR, 'integrity_model.pkl'))
    MODELS_LOADED = True
except FileNotFoundError:
    MODELS_LOADED = False


# --- APP UI ---
st.set_page_config(layout="wide", page_title="Behavioral Observation Tool")
st.title("Behavioral Observation Tool")


if not MODELS_LOADED:
    st.error("One or more model files (`.pkl`) are missing. Please run all training scripts first (`generate...`, `train...`).")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Observer Inputs")

    # --- 1. ANXIETY ---
    with st.expander("1. Anxiety Estimation Inputs", expanded=True):
        st.info("Log behaviors related to nervousness and stress.")
        c1, c2 = st.columns(2)
        with c1:
            restlessness = st.checkbox("Restlessness (fidgeting, pacing)", key="anx1")
            facial_strain = st.checkbox("Facial Strain (tight jaw, furrowed brow)", key="anx2")
            multitasking = st.checkbox("Distracted / Multitasking", key="anx3")
            speech_speed = st.selectbox("Speech Speed", options=list(speech_speed_classes), index=1, key="anx4")
        with c2:
            eye_contact_breaks = st.slider("Eye Contact Breaks (per minute)", 0, 40, 5, key="anx5")
            hours_of_sleep = st.slider("Reported Hours of Sleep (last night)", 0.0, 12.0, 7.5, 0.5, key="anx6")
            caffeine_intake = st.number_input("Reported Caffeine Intake (e.g., cups)", 0, 10, 1, key="anx7")

    # --- 2. STABILITY ---
    with st.expander("2. Cognitive Stability Inputs", expanded=True):
        st.info("Assess clarity of thought and emotional regulation.")
        topic_drift = st.checkbox("Topic Drift (loses conversation track)", key="stb1")
        logical_confusion = st.checkbox("Logical Confusion (contradictory reasoning)", key="stb2")
        overwhelmed_by_tasks = st.checkbox("Overwhelmed by Small Tasks", key="stb3")
        mood_shifts = st.checkbox("Sudden Mood Shifts (e.g., calm to irritated)", key="stb4")

    # --- 3. INTEGRITY ---
    with st.expander("3. Integrity / Honesty Cross-Check Inputs", expanded=True):
        st.info("Check for consistency and signs of deception.")
        contradiction = st.checkbox("Contradiction in Story", key="int1")
        timeline_inconsistency = st.checkbox("Timeline Inconsistency", key="int2")
        cognitive_pauses = st.checkbox("Unusual Cognitive Pauses", key="int3")
        over_rehearsed_responses = st.checkbox("Overly Rehearsed / Scripted Responses", key="int4")
        stress_smiles = st.checkbox("Stress Smiles or Inappropriate Laughter", key="int5")
        body_language_contradiction = st.checkbox("Body Language Contradicts Statement", key="int6")
        
    # --- 4. OBSERVER FEELING ---
    with st.expander("4. Observer's Final Gut Feeling", expanded=True):
        st.info("Your overall intuitive assessment.")
        observer_gut_feeling = st.radio("Observer Gut Feeling", ('Good', 'Neutral', 'Bad'), index=1, key="obs1")

with col2:
    st.header("Analysis Results")
    
    if st.button("▶️ Run Analysis"):
        # --- ✨ FEATURE ENGINEERING ON THE FLY ✨ ---
        # 1. Anxiety Feature
        sleep_caffeine_interaction = (hours_of_sleep + 1) / (caffeine_intake + 1)
        
        # 2. Stability Feature
        instability_symptom_count = sum([topic_drift, logical_confusion, overwhelmed_by_tasks, mood_shifts])
        
        # 3. Integrity Feature
        contradiction_and_pause = int(contradiction and cognitive_pauses)
        
        # --- 1. ANXIETY PREDICTION ---
        speech_speed_encoded = list(speech_speed_classes).index(speech_speed)
        anxiety_input = np.array([[
            int(restlessness), speech_speed_encoded, eye_contact_breaks, 
            int(facial_strain), int(multitasking), hours_of_sleep, caffeine_intake,
            sleep_caffeine_interaction # ✨ Add new feature
        ]])
        anxiety_score = anxiety_model.predict(anxiety_input)[0]
        anxiety_score = np.clip(anxiety_score, 0, 10)
        
        anxiety_label = "Calm"
        if 4 <= anxiety_score < 7: anxiety_label = "Moderate"
        elif anxiety_score >= 7: anxiety_label = "High"
        anxiety_reason = "signs of nervous energy" if anxiety_score >= 7 else "moderate stress indicators" if anxiety_score >= 4 else "appears calm"
        st.subheader("Anxiety Score")
        st.markdown(f"### `{anxiety_score:.1f} / 10.0` → **{anxiety_label}** (*{anxiety_reason}*)")

        # --- 2. STABILITY PREDICTION ---
        stability_input = np.array([[
            int(topic_drift), int(logical_confusion), int(overwhelmed_by_tasks), int(mood_shifts),
            instability_symptom_count # ✨ Add new feature
        ]])
        stability_prediction = stability_model.predict(stability_input)[0]
        stability_reason_parts = [flag for flag, present in zip(["topic drift", "logical confusion", "being overwhelmed", "mood shifts"], [topic_drift, logical_confusion, overwhelmed_by_tasks, mood_shifts]) if present]
        stability_reason = ", ".join(stability_reason_parts) if stability_reason_parts else "no major signs of instability"
        st.subheader("Cognitive Stability")
        st.markdown(f"### **{stability_prediction}** → *Reason: {stability_reason}*")

        # --- 3. INTEGRITY PREDICTION ---
        integrity_input_data = {
            'contradiction': int(contradiction), 'timeline_inconsistency': int(timeline_inconsistency),
            'cognitive_pauses': int(cognitive_pauses), 'over_rehearsed_responses': int(over_rehearsed_responses),
            'stress_smiles': int(stress_smiles), 'body_language_contradiction': int(body_language_contradiction),
            'contradiction_and_pause': contradiction_and_pause # ✨ Add new feature
        }
        integrity_input = np.array([list(integrity_input_data.values())])
        integrity_proba = integrity_model.predict_proba(integrity_input)[0][1]
        integrity_score = int(integrity_proba * 100)
        
        integrity_label = "High"
        if 40 <= integrity_score < 70: integrity_label = "Medium"
        elif integrity_score < 40: integrity_label = "Low"
        st.subheader("Integrity / Honesty Score")
        st.markdown(f"### `{integrity_score}%` → **{integrity_label}**")
        
        red_flags = [flag.replace("_", " ").title() for flag, value in integrity_input_data.items() if value and flag != 'contradiction_and_pause']
        if contradiction_and_pause: red_flags.append("Contradiction And Cognitive Pause")
        if red_flags:
            st.error("**🔴 Red Flags Detected:**")
            for flag in red_flags: st.markdown(f"- {flag}")

        # --- 4. WORK-LIFE COMPATIBILITY ---
        st.subheader("Work-Life Compatibility")
        compatibility_label, summary = get_work_life_compatibility(anxiety_score, stability_prediction, integrity_score, observer_gut_feeling)
        st.markdown(f"## {compatibility_label}")
        st.info(f"**Reasoning:** {summary}")
        
    else:
        st.info("Adjust inputs and click 'Run Analysis' for results.")

# streamlit run "e:\carr\btech cse\computer\machine learning\anxiety detector\new\behavioral_tool_app.py"        