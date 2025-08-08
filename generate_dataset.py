import pandas as pd
import numpy as np

def generate_dataset(num_rows=2500):
    """
    Generates an improved synthetic dataset for behavioral observation modeling.
    V2: Includes engineered features to improve model performance.
    """
    np.random.seed(42)
    data = []

    personas = {
        'low_risk': {'n': int(num_rows * 0.3), 'anxiety_base': 1, 'stability_prob': 0.95, 'integrity_base': 90},
        'anxious_honest': {'n': int(num_rows * 0.25), 'anxiety_base': 6, 'stability_prob': 0.85, 'integrity_base': 85},
        'deceptive': {'n': int(num_rows * 0.20), 'anxiety_base': 5, 'stability_prob': 0.70, 'integrity_base': 20},
        'volatile': {'n': int(num_rows * 0.15), 'anxiety_base': 7, 'stability_prob': 0.40, 'integrity_base': 60},
        'trauma_survivor': {'n': int(num_rows * 0.10), 'anxiety_base': 8, 'stability_prob': 0.65, 'integrity_base': 75}
    }

    for persona, props in personas.items():
        n_persona = props['n']
        for _ in range(n_persona):
            # --- Anxiety Inputs ---
            restlessness = np.random.uniform(0, 1) < (0.1 + props['anxiety_base'] * 0.1)
            speech_speed = np.random.choice(['slow', 'normal', 'fast'], p=[max(0.05, 0.4 - props['anxiety_base']*0.04), max(0.05, 0.5 - props['anxiety_base']*0.02), max(0.05, 0.1 + props['anxiety_base']*0.06)])
            eye_contact_breaks = np.random.randint(0, 12) + props['anxiety_base'] * 1.5
            facial_strain = np.random.uniform(0, 1) < (0.05 + props['anxiety_base'] * 0.11)
            multitasking = np.random.uniform(0, 1) < (0.1 + props['anxiety_base'] * 0.06)
            hours_of_sleep = max(3, 8 - props['anxiety_base'] * 0.6 + np.random.normal(0, 0.8))
            caffeine_intake = max(0, np.random.randint(0, 4) + (props['anxiety_base'] // 2.5))
            
            # --- Stability Inputs ---
            topic_drift = np.random.uniform(0, 1) > props['stability_prob']
            logical_confusion = np.random.uniform(0, 1) > (props['stability_prob'] + 0.05)
            overwhelmed_by_tasks = np.random.uniform(0, 1) > (props['stability_prob'] + 0.1)
            mood_shifts = np.random.uniform(0, 1) > props['stability_prob']

            # --- Integrity Inputs ---
            base_integrity_prob = props['integrity_base'] / 100.0
            contradiction = np.random.uniform(0, 1) > base_integrity_prob
            timeline_inconsistency = np.random.uniform(0, 1) > (base_integrity_prob + 0.05)
            cognitive_pauses = np.random.uniform(0, 1) > (base_integrity_prob + 0.1)
            over_rehearsed_responses = np.random.uniform(0, 1) > (base_integrity_prob + 0.2)
            stress_smiles = np.random.uniform(0, 1) > (base_integrity_prob + 0.15)
            body_language_contradiction = np.random.uniform(0, 1) > base_integrity_prob

            # --- ✨ New Engineered Features ✨ ---
            # 1. For Anxiety: Interaction between sleep and caffeine
            sleep_caffeine_interaction = (hours_of_sleep + 1) / (caffeine_intake + 1)
            
            # 2. For Stability: A count of instability symptoms
            instability_symptom_count = sum([topic_drift, logical_confusion, overwhelmed_by_tasks, mood_shifts])
            
            # 3. For Integrity: A powerful combined flag
            contradiction_and_pause = int(contradiction and cognitive_pauses)

            # --- Targets ---
            anxiety_score = min(10, max(0, props['anxiety_base'] + (4 / (sleep_caffeine_interaction + 0.5)) + np.random.normal(0, 1)))
            stability_label = 'Stable' if instability_symptom_count < 2 and np.random.rand() < props['stability_prob'] else 'Unstable'
            integrity_score = min(100, max(0, props['integrity_base'] - (contradiction_and_pause * 20) + np.random.normal(0, 8)))
            
            # Observer Gut Feeling
            p_good = base_integrity_prob * props['stability_prob']
            p_bad = 1 - p_good
            gut_feeling = np.random.choice(['Good', 'Bad'], p=[p_good, p_bad])


            row = {
                'restlessness': int(restlessness),
                'speech_speed': speech_speed,
                'eye_contact_breaks': eye_contact_breaks,
                'facial_strain': int(facial_strain),
                'multitasking': int(multitasking),
                'hours_of_sleep': hours_of_sleep,
                'caffeine_intake': caffeine_intake,
                'sleep_caffeine_interaction': sleep_caffeine_interaction, # ✨ New
                'anxiety_score_target': anxiety_score,
                
                'topic_drift': int(topic_drift),
                'logical_confusion': int(logical_confusion),
                'overwhelmed_by_tasks': int(overwhelmed_by_tasks),
                'mood_shifts': int(mood_shifts),
                'instability_symptom_count': instability_symptom_count, # ✨ New
                'stability_target': stability_label,

                'contradiction': int(contradiction),
                'timeline_inconsistency': int(timeline_inconsistency),
                'cognitive_pauses': int(cognitive_pauses),
                'over_rehearsed_responses': int(over_rehearsed_responses),
                'stress_smiles': int(stress_smiles),
                'body_language_contradiction': int(body_language_contradiction),
                'contradiction_and_pause': contradiction_and_pause, # ✨ New
                'integrity_score_target': integrity_score,
                
                'observer_gut_feeling': gut_feeling,
                'persona': persona
            }
            data.append(row)

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == '__main__':
    dataset = generate_dataset()
    dataset.to_csv('synthetic_behavioral_data_v2.csv', index=False)
    print("Improved synthetic dataset generated and saved to 'synthetic_behavioral_data_v2.csv'")
    print(dataset.head())