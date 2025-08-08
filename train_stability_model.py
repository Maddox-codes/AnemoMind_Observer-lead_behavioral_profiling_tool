import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

def train_stability_model():
    """
    Loads the improved dataset and trains a hyperparameter-tuned
    RandomForestClassifier for cognitive stability.
    """
    # Load dataset
    df = pd.read_csv(r'E:\carr\btech cse\computer\machine learning\anxiety detector\new\synthetic_behavioral_data_v2.csv')

    # Define features (including the new aggregated feature) and target
    features = [
        'topic_drift', 'logical_confusion', 'overwhelmed_by_tasks', 
        'mood_shifts', 'instability_symptom_count' # ✨ New feature included
    ]
    target = 'stability_target'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- ✨ Hyperparameter Tuning using GridSearchCV ✨ ---
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'criterion': ['gini', 'entropy']
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    
    print("Starting hyperparameter tuning for Stability Model...")
    grid_search.fit(X_train, y_train)
    
    print("Best parameters found: ", grid_search.best_params_)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate
    score = best_model.score(X_test, y_test)
    print(f"IMPROVED Stability Model Accuracy: {score:.4f}")

    # Save model
    joblib.dump(best_model, 'stability_model.pkl')
    print("Improved stability model saved to 'stability_model.pkl'")

if __name__ == '__main__':
    train_stability_model()