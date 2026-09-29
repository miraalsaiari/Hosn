import csv
import random

rows = []

# -----------------------------
# 50 STAY examples
# -----------------------------
for i in range(50):

    current_rssi = random.randint(-60, -45)
    candidate_rssi = random.randint(-80, -55)

    current_latency = random.randint(15, 40)
    candidate_latency = random.randint(30, 80)

    current_loss = round(random.uniform(0, 2), 1)
    candidate_loss = round(random.uniform(1, 5), 1)

    current_trend = random.randint(-1, 5)
    candidate_trend = random.randint(-5, 2)

    rows.append([
        current_rssi,
        candidate_rssi,
        current_latency,
        candidate_latency,
        current_loss,
        candidate_loss,
        current_trend,
        candidate_trend,
        "STAY"
    ])


# -----------------------------
# 50 HANDOVER examples
# -----------------------------
for i in range(50):

    current_rssi = random.randint(-80, -65)
    candidate_rssi = random.randint(-60, -45)

    current_latency = random.randint(60, 120)
    candidate_latency = random.randint(15, 40)

    current_loss = round(random.uniform(3, 10), 1)
    candidate_loss = round(random.uniform(0, 2), 1)

    current_trend = random.randint(-15, -3)
    candidate_trend = random.randint(3, 15)

    rows.append([
        current_rssi,
        candidate_rssi,
        current_latency,
        candidate_latency,
        current_loss,
        candidate_loss,
        current_trend,
        candidate_trend,
        "HANDOVER"
    ])


# Shuffle all rows
random.shuffle(rows)


# Save to CSV
with open("hosn_data.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "current_rssi",
        "candidate_rssi",
        "current_latency",
        "candidate_latency",
        "current_loss",
        "candidate_loss",
        "current_trend",
        "candidate_trend",
        "decision"
    ])

    writer.writerows(rows)


print("Done!")
print("Created 50 STAY and 50 HANDOVER examples.")