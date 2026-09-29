import time

# Fake signal values
ap1_signals = [-45, -50, -55, -60, -65, -70, -75]
ap2_signals = [-80, -75, -70, -65, -60, -55, -50]

# Fake latency values in milliseconds
ap1_latency = [20, 22, 25, 35, 50, 75, 110]
ap2_latency = [80, 70, 60, 45, 30, 25, 20]

# Fake packet loss percentages
ap1_loss = [0, 0, 1, 1, 3, 5, 8]
ap2_loss = [6, 5, 4, 3, 1, 1, 0]

current_ap = "AP1"

ap1_history = []
ap2_history = []

better_count = 0
CONFIRMATIONS_NEEDED = 2

print("HOSN is monitoring the networks...\n")

for i in range(len(ap1_signals)):

    ap1 = ap1_signals[i]
    ap2 = ap2_signals[i]

    latency1 = ap1_latency[i]
    latency2 = ap2_latency[i]

    loss1 = ap1_loss[i]
    loss2 = ap2_loss[i]

    ap1_history.append(ap1)
    ap2_history.append(ap2)

    print("------------------------------")
    print("Current connection:", current_ap)

    print("\nAP1")
    print("Signal:", ap1, "dBm")
    print("Latency:", latency1, "ms")
    print("Packet loss:", loss1, "%")

    print("\nAP2")
    print("Signal:", ap2, "dBm")
    print("Latency:", latency2, "ms")
    print("Packet loss:", loss2, "%")

    if len(ap1_history) >= 3:

        recent_ap1 = ap1_history[-3:]
        recent_ap2 = ap2_history[-3:]

        ap1_trend = recent_ap1[-1] - recent_ap1[0]
        ap2_trend = recent_ap2[-1] - recent_ap2[0]

        ap1_getting_weaker = ap1_trend < 0
        ap2_getting_stronger = ap2_trend > 0

        ap2_better_signal = ap2 > ap1
        ap2_better_latency = latency2 < latency1
        ap2_better_loss = loss2 < loss1

        print("\nHOSN ANALYSIS")

        if ap1_getting_weaker:
            print("AP1 signal is getting weaker.")

        if ap2_getting_stronger:
            print("AP2 signal is getting stronger.")

        if current_ap == "AP1":

            if (
                ap1_getting_weaker
                and ap2_getting_stronger
                and ap2_better_signal
                and ap2_better_latency
                and ap2_better_loss
            ):
                better_count += 1

                print("AP2 is healthier than AP1.")
                print(
                    "Confirmation:",
                    better_count,
                    "/",
                    CONFIRMATIONS_NEEDED
                )

            else:
                better_count = 0
                print("Stay on AP1 for now.")

            if better_count >= CONFIRMATIONS_NEEDED:
                print("\nHOSN DECISION:")
                print("HANDOVER AP1 -> AP2\n")
                current_ap = "AP2"

    print()

    time.sleep(1)

print("Simulation finished.")
print("Final connection:", current_ap)