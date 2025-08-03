import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score

# Załaduj dane i model jak wcześniej
# model = joblib.load(...)
# X, y - Twoje dane

y_proba = model.predict_proba(X)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y, y_proba)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

# Znajdź najlepszy próg dla maksymalnego F1
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print(f"Best threshold by F1: {best_threshold:.3f}, F1: {f1_scores[best_index]:.3f}")

# Wykres
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.plot(thresholds, f1_scores[:-1], label='F1 Score')
plt.xlabel("Threshold")
plt.legend()
plt.show()
