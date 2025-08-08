import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE # Optional: for class imbalance
import joblib

def train_integrity_model():
    """
    Loads dataset, adds engineered feature, and trains a VotingClassifier for integrity.
    """
    df = pd.read_csv(r'E:\carr\btech cse\computer\machine learning\anxiety detector\new\synthetic_behavioral_data_v2.csv')
    df['integrity_label'] = df['integrity_score_target'].apply(lambda x: 1 if x > 60 else 0)

    features = [
        'contradiction', 'timeline_inconsistency', 'cognitive_pauses',
        'over_rehearsed_responses', 'stress_smiles', 'body_language_contradiction',
        'contradiction_and_pause' # ✨ New feature included
    ]
    target = 'integrity_label'

    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Optional: Further improvement with SMOTE for class imbalance ---
    # If the classes are heavily imbalanced, SMOTE can create synthetic data.
    # print("Class distribution before SMOTE:", y_train.value_counts())
    # smote = SMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)
    # print("Class distribution after SMOTE:", y_train.value_counts())
    # --------------------------------------------------------------------

    clf1 = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=20)
    clf2 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    clf3 = LogisticRegression(random_state=42, C=0.5, solver='liblinear')

    eclf1 = VotingClassifier(
        estimators=[('rf', clf1), ('xgb', clf2), ('lr', clf3)],
        voting='soft',
        weights=[0.4, 0.4, 0.2] # Giving more weight to tree-based models
    )
    eclf1 = eclf1.fit(X_train, y_train)

    score = eclf1.score(X_test, y_test)
    print(f"IMPROVED Integrity Model Accuracy: {score:.4f}")

    joblib.dump(eclf1, 'integrity_model.pkl')
    print("Improved integrity model saved to 'integrity_model.pkl'")

if __name__ == '__main__':
    train_integrity_model()