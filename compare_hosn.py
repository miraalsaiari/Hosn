import joblib
from hosn_engine import hosn_decision

# Load AI model
model = joblib.load("hosn_ai_model.pkl")


# Different network situations
test_cases = [
    {
        "name": "Very bad current AP",
        "current_rssi": -75,
        "candidate_rssi": -50,
        "current_latency": 90,
        "candidate_latency": 25,
        "current_loss": 6,
        "candidate_loss": 1,
        "current_trend": -10,
        "candidate_trend": 8
    },

    {
        "name": "Current AP is healthy",
        "current_rssi": -48,
        "candidate_rssi": -65,
        "current_latency": 20,
        "candidate_latency": 45,
        "current_loss": 0.5,
        "candidate_loss": 2,
        "current_trend": 1,
        "candidate_trend": -2
    },

    {
        "name": "Candidate has stronger signal but worse latency",
        "current_rssi": -62,
        "candidate_rssi": -52,
        "current_latency": 25,
        "candidate_latency": 70,
        "current_loss": 1,
        "candidate_loss": 1.5,
        "current_trend": -4,
        "candidate_trend": 5
    },

    {
        "name": "Small difference between networks",
        "current_rssi": -63,
        "candidate_rssi": -58,
        "current_latency": 40,
        "candidate_latency": 35,
        "current_loss": 2,
        "candidate_loss": 1.5,
        "current_trend": -3,
        "candidate_trend": 3
    }
]


for case in test_cases:

    print("\n==============================")
    print(case["name"])
    print("==============================")

    # RULE-BASED HOSN
    rule_decision, reasons = hosn_decision(
        case["current_rssi"],
        case["candidate_rssi"],
        case["current_latency"],
        case["candidate_latency"],
        case["current_loss"],
        case["candidate_loss"],
        case["current_trend"],
        case["candidate_trend"]
    )

    # AI HOSN
    features = [[
        case["current_rssi"],
        case["candidate_rssi"],
        case["current_latency"],
        case["candidate_latency"],
        case["current_loss"],
        case["candidate_loss"],
        case["current_trend"],
        case["candidate_trend"]
    ]]

    ai_decision = model.predict(features)[0]

    print("Rule-based decision:", rule_decision)
    print("AI decision:", ai_decision)

    if rule_decision == ai_decision:
        print("RESULT: They agree")
    else:
        print("RESULT: They disagree")

    print("\nRule-based reasons:")
    for reason in reasons:
        print("-", reason)