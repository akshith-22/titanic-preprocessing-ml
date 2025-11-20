"""Train multiple classifiers on the Titanic dataset.
Expects a CSV at data/titanic.csv with standard Kaggle columns.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def load_data(path="data/titanic.csv"):
    df = pd.read_csv(path)
    return df

def build_preprocessor(df):
    numeric = ["Age","Fare","SibSp","Parch"]
    categorical = ["Sex","Embarked","Pclass"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), categorical),
    ], remainder="drop")
    return pre, numeric, categorical

def prepare(df):
    y = df["Survived"]
    X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]].copy()
    X["Sex"] = X["Sex"].map({"male":0,"female":1})
    return X, y

def main():
    df = load_data()
    X, y = prepare(df)
    pre, num, cat = build_preprocessor(df)

    models = {
        "SVM": SVC(gamma="auto"),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "LogReg": LogisticRegression(max_iter=200),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "NaiveBayes": GaussianNB(),
        "Perceptron": Perceptron(),
        "SGD": SGDClassifier(max_iter=1000, tol=1e-3),
        "LinearSVC": LinearSVC(),
        "DecisionTree": DecisionTreeClassifier(random_state=42)
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("clf", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results[name] = acc

    print("Accuracy (holdout 20%):")
    for k in sorted(results):
        v = results[k]
        print(f"{k:12s} : {v*100:5.2f}%")

if __name__ == "__main__":
    main()
