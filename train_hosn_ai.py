import csv
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# --------------------------------
# 1. Load the dataset
# --------------------------------

X = []
y = []

with open("hosn_data.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:

        features = [
            float(row["current_rssi"]),
            float(row["candidate_rssi"]),
            float(row["current_latency"]),
            float(row["candidate_latency"]),
            float(row["current_loss"]),
            float(row["candidate_loss"]),
            float(row["current_trend"]),
            float(row["candidate_trend"])
        ]

        X.append(features)
        y.append(row["decision"])


print("Loaded", len(X), "network situations.")


# --------------------------------
# 2. Split data
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------
# 3. Create AI model
# --------------------------------

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)


# --------------------------------
# 4. Train AI
# --------------------------------

model.fit(X_train, y_train)


# --------------------------------
# 5. Test AI
# --------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("AI accuracy:", round(accuracy * 100, 2), "%")


# --------------------------------
# 6. Save trained model
# --------------------------------

joblib.dump(
    model,
    "hosn_ai_model.pkl"
)

print("Saved model as hosn_ai_model.pkl")