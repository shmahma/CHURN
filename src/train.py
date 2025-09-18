from preprocess import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os


X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess()
print("Preprocessing terminé. Données prêtes pour l'entraînement.")


models = {
    "Logistic_Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random_Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
}


for name, model in models.items():
    model.fit(X_train, y_train)



MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
for name, model in models.items():
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"Model '{name}' done")

print("Done")
