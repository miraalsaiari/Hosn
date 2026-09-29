def hosn_decision(
    current_rssi,
    candidate_rssi,
    current_latency,
    candidate_latency,
    current_loss,
    candidate_loss,
    current_trend,
    candidate_trend
):

    RSSI_MARGIN = 8
    reasons = []

    current_is_weakening = current_trend < 0
    candidate_is_improving = candidate_trend > 0

    better_signal = candidate_rssi >= current_rssi + RSSI_MARGIN
    better_latency = candidate_latency < current_latency
    better_loss = candidate_loss < current_loss

    if (
        current_is_weakening
        and candidate_is_improving
        and better_signal
        and better_latency
        and better_loss
    ):
        decision = "HANDOVER"

        reasons.append("Current AP signal is getting weaker")
        reasons.append("Candidate AP signal is improving")
        reasons.append("Candidate AP has significantly better signal")
        reasons.append("Candidate AP has lower latency")
        reasons.append("Candidate AP has lower packet loss")

    else:
        decision = "STAY"

        if not current_is_weakening:
            reasons.append("Current AP is not getting weaker")

        if not better_signal:
            reasons.append("Candidate AP does not have a significantly better signal")

        if not better_latency:
            reasons.append("Candidate AP does not have better latency")

        if not better_loss:
            reasons.append("Candidate AP does not have lower packet loss")

    return decision, reasons