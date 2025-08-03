import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, f1_score

# Wczytanie danych
df = pd.read_csv("data/processed_data.csv")
X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

# Wczytanie modelu
model = joblib.load("models/final_xgboost_model.joblib")

# Predykcje probabilistyczne
y_proba = model.predict_proba(X)[:, 1]

# Znajdowanie najlepszego progu wg F1
fpr, tpr, thresholds = roc_curve(y, y_proba)
f1_scores = []

for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    f1_scores.append(f1_score(y, y_pred_thresh))

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Best threshold by F1: {best_threshold:.3f}, F1: {best_f1:.3f}")

# Predykcje z najlepszym progiem
y_pred_new = (y_proba >= best_threshold).astype(int)

# Raport i macierz pomyłek dla nowego progu
print("\n=== Classification Report (custom threshold) ===")
print(classification_report(y, y_pred_new))

cm = confusion_matrix(y, y_pred_new)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix (threshold={best_threshold:.3f})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("figures/confusion_matrix_custom_threshold.png")
plt.close()

# Wykres ROC
roc_auc = roc_auc_score(y, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/roc_curve.png")
plt.close()

print(cm)