#!/usr/bin/env python3
"""Software-only router that keeps the three HOSN evaluation arms separate.

This file does not run Mininet, execute handover, load/train a model, or write
dataset rows. The live comparison harness will call route_decision(), then pass
an authorized HANDOVER to the same make-before-break executor for every arm.

Strategies:
  RSSI_BASELINE    conventional threshold/confirmation; no HOSN rules or AI
  HOSN_RULES_ONLY normalized heterogeneous rules; conflicts HOLD without AI
  HOSN_FULL_AI     same rules first; only complete conflicts may call AI

Run: python3 hosn_strategy_router.py --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import hosn_heterogeneous_controller as heterogeneous


RSSI_BASELINE = "RSSI_BASELINE"
HOSN_RULES_ONLY = "HOSN_RULES_ONLY"
HOSN_FULL_AI = "HOSN_FULL_AI"
STRATEGIES = (RSSI_BASELINE, HOSN_RULES_ONLY, HOSN_FULL_AI)


@dataclass
class BaselineState:
    threshold_dbm: float = -75.0
    required_confirmations: int = 2
    consecutive_below_threshold: int = 0

    def __post_init__(self) -> None:
        if self.threshold_dbm >= 0:
            raise ValueError("Baseline threshold must be a negative dBm value.")
        if (isinstance(self.required_confirmations, bool)
                or not isinstance(self.required_confirmations, int)
                or self.required_confirmations < 1):
            raise ValueError("required_confirmations must be a positive integer.")

    def observe(self, wifi_rssi_dbm: float) -> bool:
        if wifi_rssi_dbm <= self.threshold_dbm:
            self.consecutive_below_threshold += 1
        else:
            self.consecutive_below_threshold = 0
        return self.consecutive_below_threshold >= self.required_confirmations


@dataclass(frozen=True)
class StrategyDecision:
    strategy: str
    decision: str
    source: str
    status: str
    reason: str
    handover_authorized: bool
    ai_called: bool
    controller_result: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def route_decision(
    strategy: str,
    current: heterogeneous.AccessObservation,
    candidate: heterogeneous.AccessObservation,
    *,
    baseline_state: Optional[BaselineState] = None,
    ai_model: Optional[heterogeneous.Predictor] = None,
    config: heterogeneous.RuleConfig = heterogeneous.DEFAULT_CONFIG,
) -> StrategyDecision:
    """Route one observation to exactly one strategy; no network side effects."""
    if strategy not in STRATEGIES:
        raise ValueError("Unknown strategy: " + str(strategy))

    if strategy == RSSI_BASELINE:
        if ai_model is not None:
            raise ValueError("RSSI_BASELINE must not receive an AI model.")
        if baseline_state is None:
            raise ValueError("RSSI_BASELINE requires an explicit BaselineState.")
        if current.signal_dbm is None:
            return StrategyDecision(
                strategy, "STAY", "HOLD", "NEEDS_DATA",
                "Current Wi-Fi RSSI is missing.", False, False, None,
            )
        switch = baseline_state.observe(float(current.signal_dbm))
        action = "HANDOVER" if switch else "STAY"
        reason = (
            "RSSI threshold confirmed."
            if switch else "RSSI threshold has not received enough confirmations."
        )
        return StrategyDecision(
            strategy, action, RSSI_BASELINE,
            "THRESHOLD_CONFIRMED" if switch else "WAITING_CONFIRMATION",
            reason, switch, False, None,
        )

    if baseline_state is not None:
        raise ValueError("HOSN strategies must not receive BaselineState.")

    if strategy == HOSN_RULES_ONLY and ai_model is not None:
        raise ValueError("HOSN_RULES_ONLY must not receive an AI model.")

    model = ai_model if strategy == HOSN_FULL_AI else None
    result = heterogeneous.decide(
        current, candidate, ai_model=model, config=config
    )

    # Rules-only deliberately keeps conflicts on HOLD. Full HOSN may also HOLD
    # when its reviewed model is absent, incompatible, or fails.
    authorized = result.decision == "HANDOVER" and result.source in ("RULES", "AI")
    return StrategyDecision(
        strategy=strategy,
        decision=result.decision,
        source=result.source,
        status=result.status,
        reason=result.reason,
        handover_authorized=authorized,
        ai_called=result.ai_called,
        controller_result=result.to_dict(),
    )


class _TestOnlyModel:
    """Contract fixture, never a trained model and never used outside tests."""

    hosn_schema_version = heterogeneous.MODEL_SCHEMA_VERSION
    feature_order = heterogeneous.AI_FEATURE_ORDER
    training_data_kind = heterogeneous.REQUIRED_TRAINING_DATA_KIND

    def __init__(self, label="HANDOVER", error=None):
        self.label = label
        self.error = error
        self.calls = []

    def predict(self, rows):
        self.calls.append(rows)
        if self.error is not None:
            raise self.error
        return [self.label]


def _clear_observations():
    current = heterogeneous.AccessObservation(
        access_name="wifi", technology="wifi_rssi",
        signal_dbm=-75.0, service_floor_dbm=-80.0,
        latency_ms=25.0, loss_pct=1.0, trend_db_per_s=-1.0,
    )
    candidate = heterogeneous.AccessObservation(
        access_name="emulated_5g", technology="cellular_rsrp",
        signal_dbm=-90.0, service_floor_dbm=-105.0,
        latency_ms=20.0, loss_pct=0.0, trend_db_per_s=0.0,
    )
    return current, candidate


def _conflict_observations():
    current, candidate = _clear_observations()
    candidate = heterogeneous.AccessObservation(
        **dict(asdict(candidate), latency_ms=45.0, loss_pct=2.0)
    )
    return current, candidate


def run_self_tests() -> int:
    import unittest

    class Tests(unittest.TestCase):
        def test_three_strategies_are_distinct(self):
            self.assertEqual(
                STRATEGIES,
                (RSSI_BASELINE, HOSN_RULES_ONLY, HOSN_FULL_AI),
            )

        def test_baseline_uses_only_threshold_confirmation(self):
            current, candidate = _clear_observations()
            state = BaselineState(threshold_dbm=-74.0, required_confirmations=2)
            first = route_decision(
                RSSI_BASELINE, current, candidate, baseline_state=state
            )
            second = route_decision(
                RSSI_BASELINE, current, candidate, baseline_state=state
            )
            self.assertEqual(first.decision, "STAY")
            self.assertEqual((second.decision, second.source),
                             ("HANDOVER", RSSI_BASELINE))
            self.assertFalse(second.ai_called)

        def test_rules_only_clear_handover(self):
            current, candidate = _clear_observations()
            result = route_decision(HOSN_RULES_ONLY, current, candidate)
            self.assertEqual((result.decision, result.source),
                             ("HANDOVER", "RULES"))
            self.assertTrue(result.handover_authorized)
            self.assertFalse(result.ai_called)

        def test_rules_only_conflict_holds(self):
            current, candidate = _conflict_observations()
            result = route_decision(HOSN_RULES_ONLY, current, candidate)
            self.assertEqual((result.decision, result.source, result.status),
                             ("STAY", "HOLD", "AI_UNAVAILABLE"))
            self.assertFalse(result.handover_authorized)
            self.assertFalse(result.ai_called)

        def test_full_ai_conflict_calls_ai(self):
            current, candidate = _conflict_observations()
            model = _TestOnlyModel("HANDOVER")
            result = route_decision(
                HOSN_FULL_AI, current, candidate, ai_model=model
            )
            self.assertEqual((result.decision, result.source, result.status),
                             ("HANDOVER", "AI", "AI_DECIDED"))
            self.assertTrue(result.handover_authorized)
            self.assertTrue(result.ai_called)
            self.assertEqual(len(model.calls), 1)

        def test_full_ai_clear_case_bypasses_ai(self):
            current, candidate = _clear_observations()
            model = _TestOnlyModel(error=RuntimeError("must not be called"))
            result = route_decision(
                HOSN_FULL_AI, current, candidate, ai_model=model
            )
            self.assertEqual((result.decision, result.source),
                             ("HANDOVER", "RULES"))
            self.assertEqual(model.calls, [])

        def test_full_ai_without_model_holds_on_conflict(self):
            current, candidate = _conflict_observations()
            result = route_decision(HOSN_FULL_AI, current, candidate)
            self.assertEqual((result.decision, result.source, result.status),
                             ("STAY", "HOLD", "AI_UNAVAILABLE"))
            self.assertFalse(result.handover_authorized)

        def test_rules_only_rejects_supplied_model(self):
            current, candidate = _conflict_observations()
            model = _TestOnlyModel("HANDOVER")
            with self.assertRaises(ValueError):
                route_decision(
                    HOSN_RULES_ONLY, current, candidate, ai_model=model
                )
            self.assertEqual(model.calls, [])

        def test_wrong_state_for_strategy_is_rejected(self):
            current, candidate = _clear_observations()
            with self.assertRaises(ValueError):
                route_decision(
                    HOSN_FULL_AI, current, candidate,
                    baseline_state=BaselineState(),
                )

    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    if result.wasSuccessful():
        print("PASS: {} three-strategy routing checks.".format(result.testsRun))
        print("No Mininet run, model training, dataset, or network change occurred.")
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Software-only router for three distinct HOSN evaluation arms."
    )
    parser.add_argument("--self-test", action="store_true", required=True)
    args = parser.parse_args()
    if args.self_test:
        return run_self_tests()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
