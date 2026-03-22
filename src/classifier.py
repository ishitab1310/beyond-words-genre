from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np


def show_top_features(model, feature_names):
    coef = model.named_steps["clf"].coef_

    for i, label in enumerate(["literature", "news", "social"]):
        top = coef[i].argsort()[-10:]
        print(f"\nTop features for {label}:")
        for idx in top:
            print(feature_names[idx])


def train_classifier(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)


    show_top_features(model, feature_names)

    print("\n[CLASSIFIER RESULTS]")
    print(classification_report(y_test, preds))