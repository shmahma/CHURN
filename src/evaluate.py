from preprocess import load_and_preprocess
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess()



MODEL_DIR = "models"
models = {}
for file in os.listdir(MODEL_DIR):
    if file.endswith(".joblib"):
        name = file.replace(".joblib", "")
        path = os.path.join(MODEL_DIR, file)
        models[name] = joblib.load(path)

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Conf Matr - {name}")
    plt.xlabel("Predictions")
    plt.ylabel("Real")
    plt.show()

for name, model in models.items():
    evaluate_model(name, model, X_test, y_test)
