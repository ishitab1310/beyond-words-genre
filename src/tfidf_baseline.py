from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def run_tfidf_baseline(df):
    X_text = df["text"].values
    y = df["genre"].values

    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n [TF-IDF BASELINE RESULTS]")
    print(classification_report(y_test, preds))