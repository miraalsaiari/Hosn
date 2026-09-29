import joblib

# Load the trained AI model
model = joblib.load("hosn_ai_model.pkl")

# New network situation the AI has not seen before
network_case = [[
    -48,
    -65,
    20,
    45,
    0.2,
    2.0,
    1,
    -2
]]
prediction = model.predict(network_case)

print("HOSN AI DECISION:", prediction[0])