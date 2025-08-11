from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

data = pd.read_csv("breast-cancer.csv")

data = data.drop(["id"], axis=1)

data["diagnosis"] = data["diagnosis"].map({"M":1, "B":0})

X = data.drop(["diagnosis"], axis=1)
y = data["diagnosis"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=20)

model = LogisticRegression()

scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

print("Accuracy for each fold :", scores)
print("Mean Accuracy:", scores.mean())

model.fit(X_scaled, y)

joblib.dump(model, "breast_cancer_model.joblib")
joblib.dump(scaler, "scaler.joblib")