import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("data/processed_data.csv")

X = df.iloc[:,:-1].to_numpy()
y = df.iloc[:,-1].to_numpy()

scale_pos_weight_value = (y == 0).sum() / (y == 1).sum()

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', max(1, scale_pos_weight_value*0.8), scale_pos_weight_value*1.2),
        'eval_metric': 'logloss',
        'random_state': 42
    }

    model = XGBClassifier(**param)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring='recall', cv=cv)

    return scores.mean()
    
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

best_params = study.best_trial.params
final_model = XGBClassifier(**best_params, eval_metric='logloss', random_state=42)
final_model.fit(X, y)

joblib.dump(final_model, "models/final_xgboost_model.joblib")
