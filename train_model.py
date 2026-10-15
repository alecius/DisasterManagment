import os
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("disaster_data.csv")

features = [
    "magnitude",
    "people_affected",
    "buildings_damaged",
    "infrastructure_damage"
]

X = data[features]


# MODEL 1: DAMAGE PREDICTION

y_damage = data["damage_severity"]

damage_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

damage_model.fit(X, y_damage)

joblib.dump(
    damage_model,
    "models/damage_model.pkl"
)

print("Damage model trained successfully!")


# MODEL 2: DISASTER REPORT CLASSIFICATION

reports = [
    "building collapsed earthquake",
    "severe flood destroyed houses",
    "minor cracks in building",
    "road blocked landslide",
    "heavy rainfall flooding",
    "small damage to house",
    "people trapped collapsed building",
    "forest fire spreading",
    "minor road damage",
    "bridge destroyed flood",
    "water entered houses",
    "severe structural damage"
]

report_labels = [
    "Severe",
    "Severe",
    "Minor",
    "Moderate",
    "Moderate",
    "Minor",
    "Severe",
    "Severe",
    "Minor",
    "Severe",
    "Moderate",
    "Severe"
]

report_model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

report_model.fit(reports, report_labels)

joblib.dump(
    report_model,
    "models/report_model.pkl"
)

print("Report model trained successfully!")


# MODEL 3: RESPONSE PRIORITY PREDICTION

y_priority = data["response_priority"]

priority_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=4,
    random_state=42
)

priority_model.fit(X, y_priority)

joblib.dump(
    priority_model,
    "models/priority_model.pkl"
)

print("Priority model trained successfully!")

print("\nAll three models have been trained!")