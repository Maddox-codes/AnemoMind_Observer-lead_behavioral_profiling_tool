import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib

def train_anxiety_model():
    """
    Loads the improved dataset, preprocesses data, and trains a hyperparameter-tuned
    RandomForestRegressor for anxiety estimation.
    """
    # Load dataset

    df = pd.read_csv(r'E:\carr\btech cse\computer\machine learning\anxiety detector\new\synthetic_behavioral_data_v2.csv')


    # Define features (including the new engineered feature) and target
    features = [
        'restlessness', 'speech_speed', 'eye_contact_breaks', 
        'facial_strain', 'multitasking', 'hours_of_sleep', 'caffeine_intake',
        'sleep_caffeine_interaction' # ✨ New feature included
    ]
    target = 'anxiety_score_target'

    X = df[features]
    y = df[target]

    # Preprocessing
    le = LabelEncoder()
    X['speech_speed'] = le.fit_transform(X['speech_speed'])
    joblib.dump(le.classes_, 'speech_speed_classes.pkl')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- ✨ Hyperparameter Tuning using GridSearchCV ✨ ---
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')
    
    print("Starting hyperparameter tuning for Anxiety Model...")
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    
    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate on the test set
    score = best_model.score(X_test, y_test)
    print(f"IMPROVED Anxiety Model R^2 Score: {score:.4f}")

    # Save the best model
    joblib.dump(best_model, 'anxiety_model.pkl')
    print("Improved anxiety model saved to 'anxiety_model.pkl'")

if __name__ == '__main__':
    train_anxiety_model()