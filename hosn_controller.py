#!/usr/bin/env python3
"""HOSN decision controller: rules first; AI only for an unclear comparison.

Stage implemented here: choose ONE decision, without changing the network.

    from hosn_controller import decide
    result = decide(**measurements)  # Rules work with no model installed.
    # Later, after retraining and separate validation:
    result = decide(**measurements, ai_model=reviewed_model)

This module uses hosn_rules.py, NOT the superseded hosn_engine.py. It does not
import compare_hosn.py, load a .pkl file, train a model, or generate a dataset.
The old hosn_ai_model.pkl is NEVER loaded automatically.

Routing:
  CLEAR rule -> use STAY/HANDOVER, never call AI.
  CONFLICT with complete inputs + supplied model -> call predict ONCE.
  Missing/invalid input, missing model, or failed prediction -> HOLD: STAY for
    now, with an explicit status. HOLD does not mean the current AP is better.

The AI interface is compatible with the friend's eight-column training script:
model.predict([[current_rssi, candidate_rssi, current_latency,
                candidate_latency, current_loss, candidate_loss,
                current_trend, candidate_trend]])
Only a single STAY or HANDOVER label is accepted. Passing a model here is not
proof of training quality. The caller must validate it on independent measured
experiments, including the unclear cases that will actually reach the model.

Measurements must follow hosn_rules.py's units and missing-data conventions.
In particular, do not fill candidate RTT/loss using current-AP measurements;
the first handover_simulation.py does NOT collect candidate-path RTT/loss.

Before live integration, the measurement/execution loop must additionally:
- verify association, candidate reachability, comparable AP-specific probes,
  observation freshness, and consistent signal-trend observation windows;
- apply temporal confirmation/cooldown so noisy samples do not cause flapping;
- request a switch and verify its BSSID/result, rather than treating a decision
  as evidence that a switch happened;
- handle disconnection/recovery separately (this is not a recovery routine).
This module is synchronous; enforce a timeout externally for a remote predictor.

Run without sudo:
    python3 hosn_controller.py --self-test
    python3 hosn_controller.py --demo

The tests/demo use invented SOFTWARE TEST inputs and a test-only predictor.
They do not train AI, run Mininet, write data files, or prove network improvement.
"""

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Protocol, Sequence

from hosn_rules import AI_FEATURE_ORDER, DEFAULT_CONFIG, RuleConfig, evaluate_rules


class Predictor(Protocol):
    """Interface accepted from a separately trained and reviewed model."""

    def predict(self, rows: Sequence[Sequence[float]]) -> Any:
        ...


@dataclass(frozen=True)
class ControllerDecision:
    """One decision plus the reason/source needed for an experiment log."""

    decision: str       # STAY or HANDOVER; NEVER ASK_AI.
    source: str         # RULES, AI, or HOLD.
    status: str         # CLEAR, AI_DECIDED, NEEDS_DATA, AI_UNAVAILABLE, etc.
    reason: str
    rule_decision: str  # Retain ASK_AI here for audit, NOT for execution.
    rule_status: str
    ai_called: bool = False
    ai_prediction: Optional[str] = None

    @property
    def on_hold(self) -> bool:
        """No justified final preference is available; collect/fix what is missing."""
        return self.source == "HOLD"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-compatible fields for a later experiment log."""
        return asdict(self)


def decide(
    current_rssi: Optional[float],
    candidate_rssi: Optional[float],
    current_latency: Optional[float],
    candidate_latency: Optional[float],
    current_loss: Optional[float],
    candidate_loss: Optional[float],
    current_trend: Optional[float] = None,
    candidate_trend: Optional[float] = None,
    *,
    ai_model: Optional[Predictor] = None,
    config: RuleConfig = DEFAULT_CONFIG,
) -> ControllerDecision:
    """Route one comparison. No file/model loading and no network side effects.

    A supplied model is consulted ONLY for a complete CONFLICT result. In a
    valid conflict case its STAY/HANDOVER label becomes the controller decision.
    Missing/invalid data and model errors are NOT handed to an AI as if they
    were legitimate evidence. A configuration error raises before model use.
    """
    values = dict(zip(AI_FEATURE_ORDER, (
        current_rssi, candidate_rssi, current_latency, candidate_latency,
        current_loss, candidate_loss, current_trend, candidate_trend,
    )))
    rule = evaluate_rules(**values, config=config)

    def result(action, source, status, reason, called=False, prediction=None):
        return ControllerDecision(
            decision=action, source=source, status=status, reason=reason,
            rule_decision=rule.decision, rule_status=rule.status,
            ai_called=called, ai_prediction=prediction,
        )

    if rule.status == "CLEAR" and rule.decision in ("STAY", "HANDOVER"):
        return result(rule.decision, "RULES", "CLEAR", rule.reason)

    if not rule.needs_ai:
        return result("STAY", "HOLD", rule.status, rule.reason)

    if ai_model is None:
        return result(
            "STAY", "HOLD", "AI_UNAVAILABLE",
            rule.reason + " No reviewed AI model is connected; do not switch yet.",
        )

    # Rule validation has established that all eight inputs are present/finite.
    # Build precisely one row in the same order as the training script.
    features = [[float(values[name]) for name in AI_FEATURE_ORDER]]
    try:
        predict = getattr(ai_model, "predict", None)
        if not callable(predict):
            return result(
                "STAY", "HOLD", "AI_INVALID_MODEL",
                "Supplied model does not have a callable predict method.",
            )
    except Exception as exc:
        return result(
            "STAY", "HOLD", "AI_INVALID_MODEL",
            "Could not access model.predict: " + type(exc).__name__,
        )

    try:
        predictions = predict(features)
    except Exception as exc:
        return result(
            "STAY", "HOLD", "AI_ERROR",
            "AI prediction failed (" + type(exc).__name__ + "); do not switch.",
            called=True,
        )

    # Do not silently turn an empty/multirow/numeric/unknown output into an action.
    try:
        if isinstance(predictions, (str, bytes)) or len(predictions) != 1:
            raise ValueError("Expected exactly one prediction.")
        label = predictions[0]
        if not isinstance(label, str) or label not in ("STAY", "HANDOVER"):
            raise ValueError("Expected STAY or HANDOVER.")
    except Exception:
        return result(
            "STAY", "HOLD", "AI_INVALID_OUTPUT",
            "AI must return one STAY or HANDOVER label; do not switch.",
            called=True,
        )

    label = str(label)  # Normalize e.g. a numpy string scalar for JSON logs.
    return result(
        label, "AI", "AI_DECIDED",
        rule.reason + " AI selected " + label + ".",
        called=True, prediction=label,
    )


# The objects below are ONLY software-test fixtures. They are not trained AI.
class _TestOnlyPredictor:
    def __init__(self, reply=None, error=None):
        self.reply = ["HANDOVER"] if reply is None else reply
        self.error = error
        self.calls = []

    def predict(self, rows):
        self.calls.append(rows)
        if self.error is not None:
            raise self.error
        return self.reply


def _test_inputs():
    """Invented numbers, never exported or used as training data."""
    return dict(
        current_rssi=-75.0, candidate_rssi=-60.0,
        current_latency=20.0, candidate_latency=20.0,
        current_loss=0.0, candidate_loss=0.0,
        current_trend=-1.0, candidate_trend=1.0,
    )


def run_self_tests() -> int:
    """Check routing, output validation, and the no-AI-on-clear-cases guarantee."""
    base = _test_inputs()
    checks = 0

    def check(name, changes, expected, model=None, called=False, config=DEFAULT_CONFIG):
        nonlocal checks
        before = len(model.calls) if isinstance(model, _TestOnlyPredictor) else 0
        observed = decide(**dict(base, **changes), ai_model=model, config=config)
        actual = (observed.decision, observed.source, observed.status)
        if actual != expected:
            raise AssertionError(name + ": expected " + repr(expected) + ", got " + repr(actual))
        if observed.ai_called is not called:
            raise AssertionError(name + ": incorrect AI call flag")
        if isinstance(model, _TestOnlyPredictor):
            if len(model.calls) - before != int(called):
                raise AssertionError(name + ": AI was not called exactly as intended")
        if observed.on_hold != (observed.source == "HOLD"):
            raise AssertionError(name + ": incorrect hold status")
        if not observed.reason or observed.decision not in ("STAY", "HANDOVER"):
            raise AssertionError(name + ": missing reason or non-executable action")
        if observed.to_dict()["decision"] != observed.decision:
            raise AssertionError(name + ": serialization changed the action")
        checks += 1
        return observed

    switch = ("HANDOVER", "RULES", "CLEAR")
    stay = ("STAY", "RULES", "CLEAR")
    needs_data = ("STAY", "HOLD", "NEEDS_DATA")
    invalid = ("STAY", "HOLD", "INVALID_DATA")
    conflict = {"candidate_latency": 40.0}
    trap = _TestOnlyPredictor(error=RuntimeError("Must not call AI for this case"))

    check("Equal loss permits a clear switch without AI", {}, switch)
    check("Clear switch ignores even a supplied broken AI", {}, switch, trap)
    check("Clear stay without AI", {"candidate_rssi": -80.0}, stay)
    check("Clear stay never calls AI", {"candidate_rssi": -80.0}, stay, trap)
    check("All primary metrics favor candidate", {
        "candidate_latency": 10.0, "current_loss": 3.0, "candidate_loss": 1.0,
    }, switch, trap)
    check("Equal nonzero loss is allowed", {"current_loss": 2.0, "candidate_loss": 2.0}, switch, trap)
    check("Eight dB is the configured boundary", {"candidate_rssi": -67.0}, switch, trap)
    check("Below margin without quality benefit stays", {"candidate_rssi": -67.01}, stay, trap)
    check("Clear rule does not require unused trend history", {
        "current_trend": None, "candidate_trend": None,
    }, switch, trap)
    check("Conflict waits rather than loading old model", conflict,
          ("STAY", "HOLD", "AI_UNAVAILABLE"))

    ai_switch = _TestOnlyPredictor(["HANDOVER"])
    decision = check("Conflict uses AI HANDOVER once", conflict,
                     ("HANDOVER", "AI", "AI_DECIDED"), ai_switch, True)
    expected_row = [[float(dict(base, **conflict)[name]) for name in AI_FEATURE_ORDER]]
    if ai_switch.calls[0] != expected_row or decision.ai_prediction != "HANDOVER":
        raise AssertionError("AI feature order or prediction record changed")
    if decision.rule_decision != "ASK_AI" or decision.rule_status != "CONFLICT":
        raise AssertionError("Original rule conflict was not recorded")
    checks += 1
    check("Conflict uses AI STAY once", conflict, ("STAY", "AI", "AI_DECIDED"),
          _TestOnlyPredictor(["STAY"]), True)
    check("Worse candidate loss is a conflict", {"candidate_loss": 2.0},
          ("STAY", "AI", "AI_DECIDED"), _TestOnlyPredictor(["STAY"]), True)
    check("Delay improvement without signal margin is a conflict", {
        "candidate_rssi": -72.0, "candidate_latency": 10.0,
    }, ("HANDOVER", "AI", "AI_DECIDED"), _TestOnlyPredictor(["HANDOVER"]), True)

    for field in AI_FEATURE_ORDER[:6]:
        check("Missing " + field, {field: None}, needs_data, trap)
    for field in AI_FEATURE_ORDER[6:]:
        check("Conflict requires " + field, dict(conflict, **{field: None}), needs_data, trap)
    for field in AI_FEATURE_ORDER:
        check("Nonfinite " + field, {field: float("nan")}, invalid, trap)
    for changes in (
        {"current_latency": -1.0}, {"candidate_loss": 101.0},
        {"current_loss": -1.0}, {"candidate_rssi": 10.0},
        {"current_trend": True}, {"candidate_latency": "20"},
        {"current_latency": 0.0, "current_loss": 100.0},
    ):
        check("Invalid measured value " + repr(changes), changes, invalid, trap)
    check("Complete probe loss is not zero latency", {
        "candidate_latency": None, "candidate_loss": 100.0,
    }, needs_data, trap)
    check("Configured measurement tolerance is respected", {"candidate_latency": 20.5},
          switch, trap, config=RuleConfig(latency_tolerance_ms=1.0))
    check("Missing model method does not produce handover", conflict,
          ("STAY", "HOLD", "AI_INVALID_MODEL"), object())
    check("AI exception holds", conflict, ("STAY", "HOLD", "AI_ERROR"),
          _TestOnlyPredictor(error=RuntimeError("Test-only failure")), True)
    for reply in ([], ["STAY", "HANDOVER"], ["ASK_AI"], ["handover"],
                  [True], [1], [None], [["HANDOVER"]], "HANDOVER", b"STAY", 42):
        check("Invalid AI reply " + repr(reply), conflict,
              ("STAY", "HOLD", "AI_INVALID_OUTPUT"), _TestOnlyPredictor(reply), True)
    try:
        decide(**base, ai_model=trap, config=None)
    except TypeError:
        checks += 1
    else:
        raise AssertionError("Invalid controller configuration did not raise")
    if trap.calls:
        raise AssertionError("An AI call escaped the routing checks")

    print("PASS: {} controller/routing checks.".format(checks))
    print("Tests used invented SOFTWARE TEST inputs and test-only predictors, not trained AI.")
    print("No old model was loaded, no network was run, and no experiment files were changed.")
    return 0


def run_demo() -> int:
    base = _test_inputs()
    print("SOFTWARE DEMO ONLY: invented examples, not measurements or a trained AI.\n")
    cases = (
        ("Clear improvement", base, None),
        ("No worthwhile improvement", dict(base, candidate_rssi=-80.0), None),
        ("Conflicting measurements; AI not connected", dict(base, candidate_latency=40.0), None),
        ("Conflict; TEST-ONLY predictor says HANDOVER", dict(base, candidate_latency=40.0),
         _TestOnlyPredictor(["HANDOVER"])),
    )
    for name, measurements, model in cases:
        decision = decide(**measurements, ai_model=model)
        print(name)
        print("  {} | source={} | status={} | AI called={}".format(
            decision.decision, decision.source, decision.status, decision.ai_called))
        print("  " + decision.reason + "\n")
    print("This demo did not load a model, collect data, or switch an access point.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="HOSN rules-first decision controller (no network execution yet)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true", help="Run local software routing checks")
    mode.add_argument("--demo", action="store_true", help="Show invented examples; not trained AI")
    args = parser.parse_args()
    return run_self_tests() if args.self_test else run_demo()


if __name__ == "__main__":
    raise SystemExit(main())
