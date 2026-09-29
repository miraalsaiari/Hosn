from hosn_engine import hosn_decision


decision, reasons = hosn_decision(
    current_rssi=-50,
    candidate_rssi=-58,

    current_latency=20,
    candidate_latency=35,

    current_loss=0.5,
    candidate_loss=1.5,

    current_trend=1,
    candidate_trend=-1
)


print("HOSN DECISION:", decision)

print("\nWHY?")

for reason in reasons:
    print("-", reason)