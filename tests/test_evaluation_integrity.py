import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.latent_pipeline.counterfactual_runner import EVALUATION_METHOD, _apply_cf, run as run_counterfactual
from src.latent_pipeline.evaluate_staged import _resample_patient_rows
from src.latent_pipeline.evaluate_staged import run as run_evaluation
from src.latent_pipeline.prediction_targets import add_composite_probability
from src.latent_pipeline.stage4_steward import STATE_KEYS, _update_state
from src.latent_pipeline.temporal_loop_runner import run as run_temporal_loop


class EvaluationIntegrityTest(unittest.TestCase):
    def test_patient_resampling_preserves_cluster_multiplicity(self):
        frame = pd.DataFrame(
            {
                "patient": ["a", "a", "b"],
                "value": [1, 2, 3],
            }
        )
        sampled = _resample_patient_rows(frame, frame["patient"], ["a", "a", "b"])
        self.assertEqual(len(sampled), 5)
        self.assertEqual(int((sampled["patient"] == "a").sum()), 4)

    def test_deterioration_is_max_support_composite(self):
        probs = add_composite_probability(
            {
                "vasopressor_signal": 0.2,
                "resp_support_signal": 0.6,
                "renal_support_signal": 0.1,
                "any_deterioration": 0.99,
            }
        )
        self.assertEqual(probs["any_deterioration"], 0.6)

    def test_steward_has_five_states(self):
        state = _update_state(
            None,
            {"predicted_actions": ["infection workup and antimicrobial review"]},
            {"status": "PASS"},
            ["INF_01"],
        )
        self.assertEqual(len(STATE_KEYS), 5)
        self.assertEqual(state["systemic_inflammation_state"], 1)

    def test_standardized_perturbation_changes_only_eligible_facts(self):
        facts = [{"feature": "map", "value_last": 60.0, "trend": "falling"}]
        perturbed, applied, reason = _apply_cf("map_low_to_normal", facts)
        self.assertTrue(applied)
        self.assertEqual(reason, "applied")
        self.assertEqual(perturbed[0]["value_last"], 75.0)
        self.assertEqual(facts[0]["value_last"], 60.0)

    def test_counterfactual_runner_reruns_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol_path.write_text("{}\n")
            stage4 = root / "stage4_steward.jsonl"
            stage4.write_text(
                json.dumps(
                    {
                        "example_id": "e1",
                        "source_dataset": "test",
                        "patient_id": "p1",
                        "encounter_id": "v1",
                        "packet": {
                            "facts": [{"feature": "map", "value_last": 60.0, "trend": "falling"}],
                            "latent_state": {},
                        },
                        "target_isolation_verified": True,
                        "temporal_loop": {"retry_attempts_used": 0},
                        "selected_rule_ids": [],
                        "reasoner_prediction": {
                            "any_deterioration_definition": "max_support_probability",
                            "risk_probs": {
                                "vasopressor_signal": 1.0,
                                "resp_support_signal": 0.0,
                                "renal_support_signal": 0.0,
                            }
                        },
                        "individual_protocol_state_prev": {},
                        "individual_protocol_state_next": {
                            key: 0 for key in STATE_KEYS
                        },
                    }
                )
                + "\n"
            )

            run_counterfactual(
                str(root),
                str(protocol_path),
                agent_backend="deterministic",
            )

            metrics = json.loads((root / "counterfactual_metrics.json").read_text())
            self.assertEqual(metrics["evaluation_method"], EVALUATION_METHOD)
            self.assertEqual(metrics["n_applied"], 1)

    def test_corrected_temporal_outputs_pass_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            protocol_path = root / "protocol.json"
            out_dir = root / "run"
            protocol_path.write_text("{}\n")
            input_path.write_text(
                json.dumps(
                    {
                        "source_dataset": "test",
                        "patient_id": "p1",
                        "encounter_id": "v1",
                        "step_id": 0,
                        "protocol_observations": [
                            {"feature": "map", "value_last": 60.0, "trend": "falling"}
                        ],
                        "targets": {
                            "vasopressor_signal": 1,
                            "resp_support_signal": 0,
                            "renal_support_signal": 0,
                            "any_deterioration": 1,
                        },
                    }
                )
                + "\n"
            )
            run_temporal_loop(
                str(input_path),
                str(protocol_path),
                str(out_dir),
                agent_backend="deterministic",
            )
            run_evaluation(str(out_dir), str(protocol_path), bootstrap_replicates=5)

            metrics = json.loads((out_dir / "metrics_overall.json").read_text())
            self.assertEqual(metrics["evaluation_schema"], "corrected_evaluation_v2")
            self.assertEqual(metrics["bootstrap_replicates"], 5)
            self.assertFalse((out_dir / "counterfactual_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
