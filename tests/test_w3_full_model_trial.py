import importlib.util
import json
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_qwen3_8b_w3_full_m1_trial.py"
SPEC = importlib.util.spec_from_file_location("w3_full_runner", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class W3FullModelTrialTest(unittest.TestCase):
    def test_protocol_preserves_prompts_and_freezes_graph_primary(self):
        config = json.loads(RUNNER.PROTOCOL.read_text(encoding="utf-8"))
        previous = json.loads(
            (ROOT / "configs/acceleration/qwen3_8b_full_m1_v2.json").read_text(encoding="utf-8")
        )
        RUNNER.validate_protocol(config)
        for key in ("seed", "decode_steps", "warmup", "repeats", "prompts", "max_prompt_tokens"):
            self.assertEqual(config[key], previous[key])
        self.assertEqual(config["primary_mode"], "sequence_graph")
        self.assertFalse(config["prepare_candidate"])
        self.assertNotIn("backend_acceptance_policy", config)
        self.assertEqual(
            RUNNER.sha256_file(RUNNER.PROTOCOL),
            "a82aec5904b29b391562dd45872b5955160191ce52be23d66899085fdceab127",
        )

    def test_corrected_protocol_adds_two_independent_packed_arms(self):
        config = json.loads(RUNNER.CORRECTED_PROTOCOL.read_text(encoding="utf-8"))
        previous = json.loads(RUNNER.PROTOCOL.read_text(encoding="utf-8"))
        RUNNER.validate_protocol(config)
        for key in ("seed", "decode_steps", "warmup", "repeats", "prompts", "max_prompt_tokens"):
            self.assertEqual(config[key], previous[key])
        self.assertEqual(config["packed_routes"], {
            "packed_w3_fast_corrected": "fast_corrected",
            "packed_w3_observed_exact": "observed_exact",
        })
        self.assertEqual(config["target_compute_capability"], [8, 0])
        self.assertEqual(config["target_min_vram_bytes"], 75000000000)
        self.assertEqual(RUNNER.packed_routes(config), config["packed_routes"])
        self.assertNotIn("backend_acceptance_policy", config)
        self.assertEqual(
            RUNNER.sha256_file(RUNNER.CORRECTED_PROTOCOL),
            "f7d3f12cc29a645e1d9862c4fc245a8ffc32b22428136e6b8c86427b55256b24",
        )

    def test_calibrated_protocol_versions_history_without_rewriting_it(self):
        legacy = json.loads(RUNNER.PROTOCOL.read_text(encoding="utf-8"))
        calibrated = json.loads(RUNNER.CALIBRATED_PROTOCOL.read_text(encoding="utf-8"))
        RUNNER.validate_protocol(calibrated)
        self.assertNotEqual(calibrated["id"], legacy["id"])
        for key in (
            "seed", "decode_steps", "warmup", "repeats", "prompts",
            "max_prompt_tokens", "logprob_max_abs_tolerance",
            "max_relative_timing_range", "arms", "modes", "primary_mode",
            "row_tile_by_shape", "prepare_candidate",
        ):
            self.assertEqual(calibrated[key], legacy[key])
        self.assertEqual(
            calibrated["backend_acceptance_policy"],
            RUNNER.CALIBRATED_ACCEPTANCE_POLICY,
        )
        self.assertEqual(calibrated["timing_trim_per_side"], 1)

    def test_timing_and_exact_route_guards(self):
        self.assertTrue(RUNNER.statistics_row([10.0, 10.1], 0.05)["stable"])
        self.assertFalse(RUNNER.statistics_row([10.0, 12.0], 0.05)["stable"])
        for samples in ([], [float("nan")], [0.0], [-1.0]):
            with self.assertRaises(ValueError):
                RUNNER.statistics_row(samples, 0.05)
        routes = {str(index): {"dense_fallback": 0, "packed_m1": 32} for index in range(252)}
        RUNNER.require_routes(routes, 32)
        routes["0"]["dense_fallback"] = 1
        with self.assertRaises(RuntimeError):
            RUNNER.require_routes(routes, 32)
        trace = {
            "cache_length": 5,
            "logits": torch.ones(1),
            "predictions": torch.ones(1),
            "fed_tokens": torch.ones(1),
        }
        RUNNER.exact_trace(trace, trace)
        with self.assertRaises(RuntimeError):
            RUNNER.exact_trace({**trace, "logits": torch.zeros(1)}, trace)

    def test_trimmed_stability_tolerates_one_spike_but_not_two(self):
        # Reproduces the 2026-09-17 A100 pattern: nine samples within 0.1% and a
        # single host-side spike, which the raw (max-min)/median rule voided.
        spiked = [476.7] * 9 + [558.4]
        raw = RUNNER.statistics_row(spiked, 0.05, trim_per_side=0)
        trimmed = RUNNER.statistics_row(spiked, 0.05, trim_per_side=1)
        self.assertFalse(raw["stable"])
        self.assertTrue(trimmed["stable"])
        self.assertEqual(trimmed["stability_rule"], "trimmed_relative_range")
        self.assertEqual(trimmed["trim_per_side"], 1)
        self.assertEqual(trimmed["trimmed_sample_count"], 8)
        # Median and the raw range stay reported so the spike remains auditable.
        self.assertEqual(raw["median"], trimmed["median"])
        self.assertEqual(raw["relative_range"], trimmed["relative_range"])
        # Two spikes must still fail: trimming removes one sample per side only.
        twice = [476.7] * 8 + [558.4, 557.1]
        self.assertFalse(RUNNER.statistics_row(twice, 0.05, trim_per_side=1)["stable"])
        # Genuine wide spread is not rescued by trimming.
        spread = [10.0, 10.4, 10.8, 11.2, 11.6, 12.0]
        self.assertFalse(RUNNER.statistics_row(spread, 0.05, trim_per_side=1)["stable"])
        # Too few samples fall back to the raw range instead of passing freely.
        short = RUNNER.statistics_row([10.0, 12.0], 0.05, trim_per_side=1)
        self.assertEqual(short["trim_per_side"], 0)
        self.assertEqual(short["stability_rule"], "relative_range")
        self.assertFalse(short["stable"])
        with self.assertRaises(ValueError):
            RUNNER.statistics_row([10.0] * 10, 0.05, trim_per_side=-1)

    def test_relative_backend_gate_uses_measured_amplification_control(self):
        # Measured 2026-09-16/17 A100 values; see docs/performance/W3_NUMERICAL_GATE_CALIBRATION.md.
        floor = 0.01431  # original_bf16 dynamic-vs-static, zero quantization
        structural = RUNNER.relative_backend_gate(0.01683, floor, 1.0)
        self.assertAlmostEqual(structural["ratio"], 0.01683 / floor, places=9)
        self.assertFalse(structural["passed"])
        # The absolute 0.005 limit rejects the zero-quantization control itself,
        # so the relative gate must accept anything at or below that floor.
        self.assertTrue(RUNNER.relative_backend_gate(floor, floor, 1.0)["passed"])
        self.assertTrue(RUNNER.relative_backend_gate(0.004, floor, 1.0)["passed"])
        self.assertFalse(RUNNER.relative_backend_gate(0.0144, floor, 1.0)["passed"])
        for bad in (float("nan"), float("inf"), "0.01"):
            self.assertFalse(RUNNER.relative_backend_gate(bad, floor, 1.0)["passed"])
            self.assertIsNone(RUNNER.relative_backend_gate(bad, floor, 1.0)["ratio"])
        degenerate = RUNNER.relative_backend_gate(0.001, 0.0, 1.0)
        self.assertFalse(degenerate["passed"])
        self.assertIsNone(degenerate["ratio"])
        self.assertFalse(RUNNER.relative_backend_gate(None, floor, 1.0)["passed"])
        self.assertFalse(RUNNER.relative_backend_gate(0.01, floor, 0.0)["passed"])

    def test_relative_logprob_gate_uses_same_run_control(self):
        gate = RUNNER.relative_logprob_gate(0.6, 0.7, 1.0)
        self.assertTrue(gate["passed"])
        self.assertAlmostEqual(gate["ratio"], 0.6 / 0.7)
        self.assertFalse(RUNNER.relative_logprob_gate(0.8, 0.7, 1.0)["passed"])
        for bad in (None, float("nan"), float("inf"), "0.6"):
            self.assertFalse(RUNNER.relative_logprob_gate(bad, 0.7, 1.0)["passed"])
        self.assertFalse(RUNNER.relative_logprob_gate(0.1, 0.0, 1.0)["passed"])

    def test_calibrated_backend_gate_keeps_trace_invariants_fail_closed(self):
        trace = {
            "passed": False,  # Legacy 0.005/0.05 absolute gates are report-only.
            "logits": {
                "max_abs_error": 0.4,
                "reference_rms": 25.0,
                "normalized_rmse": 0.012,
            },
            "greedy_tokens_equal": True,
            "fed_tokens_equal": True,
            "max_abs_logprob_error": 0.04,
            "logprob_tolerance": 0.05,
        }
        gate = RUNNER.calibrated_backend_gate(
            trace,
            control_nrmse=0.014,
            nrmse_limit=1.0,
            control_logprob_max_error=0.7,
            logprob_relative_limit=1.0,
        )
        self.assertTrue(gate["relative_check"]["passed"])
        self.assertTrue(gate["relative_logprob_check"]["passed"])
        self.assertTrue(gate["relative_checks_passed"])
        self.assertTrue(gate["invariants_passed"])
        self.assertTrue(gate["passed"])
        self.assertFalse(gate["legacy_absolute_check_passed"])

        for change in (
            {"greedy_tokens_equal": False},
            {"fed_tokens_equal": False},
            {"max_abs_logprob_error": 0.8},
            {"max_abs_logprob_error": float("nan")},
            {"logits": {**trace["logits"], "max_abs_error": float("inf")}},
            {"logits": {}},
        ):
            with self.subTest(change=change):
                rejected = RUNNER.calibrated_backend_gate(
                    {**trace, **change},
                    control_nrmse=0.014,
                    nrmse_limit=1.0,
                    control_logprob_max_error=0.7,
                    logprob_relative_limit=1.0,
                )
                self.assertFalse(rejected["passed"])
        self.assertFalse(
            RUNNER.calibrated_backend_gate(
                trace,
                control_nrmse=0.01,
                nrmse_limit=1.0,
                control_logprob_max_error=0.7,
                logprob_relative_limit=1.0,
            )["passed"]
        )

    def test_candidate_acceptance_uses_calibrated_gate_not_legacy_absolute_gate(self):
        def row(*, calibrated, stable=True):
            return {
                "packed_arm": "packed_w3_inline",
                "stable": stable,
                "packed_vs_decoded_w3_check": {"passed": False},
                "packed_vs_decoded_relative_check": {"passed": calibrated},
                "packed_vs_decoded_backend_acceptance": {
                    "relative_checks_passed": calibrated,
                    "invariants_passed": True,
                    "passed": calibrated,
                },
            }

        accepted = RUNNER.build_candidate_acceptance(
            [row(calibrated=True), row(calibrated=True)],
            {"packed_w3_inline": "structural"},
            RUNNER.CALIBRATED_ACCEPTANCE_POLICY,
        )["packed_w3_inline"]
        self.assertFalse(accepted["legacy_absolute_backend_checks_passed"])
        self.assertTrue(accepted["backend_checks_passed"])
        self.assertTrue(accepted["accepted"])

        rejected = RUNNER.build_candidate_acceptance(
            [row(calibrated=True), row(calibrated=False)],
            {"packed_w3_inline": "structural"},
            RUNNER.CALIBRATED_ACCEPTANCE_POLICY,
        )["packed_w3_inline"]
        self.assertFalse(rejected["backend_checks_passed"])
        self.assertFalse(rejected["accepted"])
        with self.assertRaises(RuntimeError):
            RUNNER.build_candidate_acceptance(
                [],
                {"packed_w3_inline": "structural"},
                RUNNER.CALIBRATED_ACCEPTANCE_POLICY,
            )
        legacy = RUNNER.build_candidate_acceptance(
            [row(calibrated=True), row(calibrated=True)],
            {"packed_w3_inline": "structural"},
            RUNNER.LEGACY_ACCEPTANCE_POLICY,
        )["packed_w3_inline"]
        self.assertTrue(legacy["calibrated_backend_checks_passed"])
        self.assertFalse(legacy["backend_checks_passed"])
        self.assertFalse(legacy["accepted"])

    def test_protocol_declares_gate_parameters_fail_closed(self):
        config = json.loads(RUNNER.CALIBRATED_PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(config["timing_trim_per_side"], 1)
        self.assertEqual(config["packed_vs_decoded_relative_limit"], 1.0)
        self.assertEqual(config["packed_vs_decoded_logprob_relative_limit"], 1.0)
        RUNNER.validate_protocol(config)
        for bad in ({"timing_trim_per_side": -1}, {"timing_trim_per_side": 3},
                    {"timing_trim_per_side": True}, {"timing_trim_per_side": 1.0},
                    {"packed_vs_decoded_relative_limit": 0},
                    {"packed_vs_decoded_relative_limit": -1.0},
                    {"packed_vs_decoded_relative_limit": "1"},
                    {"packed_vs_decoded_relative_limit": float("nan")},
                    {"packed_vs_decoded_relative_limit": float("inf")},
                    {"packed_vs_decoded_logprob_relative_limit": 0},
                    {"packed_vs_decoded_logprob_relative_limit": -1.0},
                    {"packed_vs_decoded_logprob_relative_limit": "1"},
                    {"packed_vs_decoded_logprob_relative_limit": float("nan")},
                    {"packed_vs_decoded_logprob_relative_limit": float("inf")},
                    {"backend_acceptance_policy": "unknown"}):
            with self.assertRaises(ValueError):
                RUNNER.validate_protocol({**config, **bad})
        # A declared trim that cannot apply must fail rather than degrade quietly.
        with self.assertRaises(ValueError):
            RUNNER.validate_protocol({**config, "repeats": 4, "timing_trim_per_side": 1})

    def test_fused_protocol_enables_nonlinear_fusion_without_touching_frozen_ones(self):
        for path in (RUNNER.PROTOCOL, RUNNER.CORRECTED_PROTOCOL):
            frozen = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("fused_nonlinear_modules", frozen)
            self.assertNotIn("backend_acceptance_policy", frozen)
            RUNNER.validate_protocol(frozen)
        fused = json.loads(RUNNER.FUSED_CALIBRATED_PROTOCOL.read_text(encoding="utf-8"))
        RUNNER.validate_protocol(fused)
        self.assertEqual(fused["fused_nonlinear_modules"], {"rms_norm": True, "rope": True})
        # A fused run must be a distinct protocol id, not a redefinition.
        base = json.loads(RUNNER.CALIBRATED_PROTOCOL.read_text(encoding="utf-8"))
        self.assertNotEqual(fused["id"], base["id"])
        for key in ("prompts", "seed", "decode_steps", "repeats", "warmup", "arms",
                    "row_tile_by_shape", "max_relative_timing_range",
                    "packed_vs_decoded_relative_limit",
                    "packed_vs_decoded_logprob_relative_limit"):
            self.assertEqual(fused[key], base[key])
        self.assertEqual(
            fused["backend_acceptance_policy"], RUNNER.CALIBRATED_ACCEPTANCE_POLICY
        )
        for bad in ({}, {"rms_norm": True}, {"rms_norm": True, "rope": 1},
                    {"rms_norm": True, "rope": True, "silu": True}):
            with self.assertRaises(ValueError):
                RUNNER.validate_protocol({**base, "fused_nonlinear_modules": bad})


if __name__ == "__main__":
    unittest.main()
