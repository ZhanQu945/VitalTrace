import json
import tempfile
import unittest
from pathlib import Path

from src.latent_pipeline.inference_context import (
    assert_no_future_fields,
    build_inference_packet,
)
from src.latent_pipeline.prompts import (
    auditor_user_prompt,
    reasoner_user_prompt,
    router_user_prompt,
    steward_user_prompt,
)
from src.latent_pipeline.stage1_router import run as run_router
from src.latent_pipeline.temporal_loop_runner import run as run_temporal_loop


class TargetIsolationTest(unittest.TestCase):
    def test_packet_builder_excludes_labels(self):
        packet = build_inference_packet(
            facts=[{"feature": "map", "value_last": 62.0}],
            latent_state={"hemodynamic_state": 2},
            individual_protocol_state_prev={"hemodynamic_state": 1},
        )
        self.assertNotIn("targets", packet)
        self.assertNotIn("risk_targets", packet)
        assert_no_future_fields(packet)

    def test_nested_future_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "risk_targets"):
            assert_no_future_fields({"latent_state": {"risk_targets": {"x": 1}}})
        with self.assertRaisesRegex(ValueError, "phase_tag"):
            assert_no_future_fields({"facts": [], "phase_tag": "intervention_window"})

    def test_prompts_reject_unsafe_packets(self):
        packet = {"facts": [], "risk_targets": {"any_deterioration": 1}}
        with self.assertRaises(ValueError):
            router_user_prompt(packet, [], max_rules=3)
        with self.assertRaises(ValueError):
            reasoner_user_prompt(packet, [], {})
        with self.assertRaises(ValueError):
            auditor_user_prompt({}, {"future_labels": {"x": 1}})
        with self.assertRaises(ValueError):
            steward_user_prompt({}, {}, {"targets": {"x": 1}}, [])

    def test_flat_router_keeps_labels_only_for_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "labeled.jsonl"
            protocol_path = root / "protocol.json"
            output_path = root / "stage1.jsonl"
            metrics_path = root / "metrics.json"
            input_path.write_text(
                json.dumps(
                    {
                        "source_dataset": "test",
                        "patient_id": "p1",
                        "encounter_id": "e1",
                        "step_id": 0,
                        "protocol_observations": [
                            {"feature": "map", "value_last": 62.0, "count": 1}
                        ],
                        "targets": {
                            "vasopressor_signal": 1,
                            "any_deterioration": 1,
                        },
                        "targets_raw": {"any_deterioration": 1},
                        "phase_tag": "intervention_window",
                    }
                )
                + "\n"
            )
            protocol_path.write_text("{}\n")

            run_router(
                str(input_path),
                str(protocol_path),
                str(output_path),
                str(metrics_path),
                agent_backend="deterministic",
            )

            row = json.loads(output_path.read_text().splitlines()[0])
            metrics = json.loads(metrics_path.read_text())
            self.assertNotIn("risk_targets", row["packet"])
            self.assertEqual(row["ground_truth_targets"]["any_deterioration"], 1)
            self.assertEqual(row["inference_context_schema"], "target_free_v1")
            self.assertTrue(row["target_isolation_verified"])
            self.assertEqual(metrics["inference_context_schema"], "target_free_v1")
            self.assertTrue(metrics["target_isolation_verified"])

    def test_temporal_loop_keeps_future_fields_outside_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "labeled.jsonl"
            protocol_path = root / "protocol.json"
            out_dir = root / "run"
            input_path.write_text(
                json.dumps(
                    {
                        "source_dataset": "test",
                        "patient_id": "p1",
                        "encounter_id": "e1",
                        "step_id": 0,
                        "protocol_observations": [
                            {"feature": "map", "value_last": 62.0, "count": 1}
                        ],
                        "targets": {"any_deterioration": 1},
                        "targets_raw": {"any_deterioration": 1},
                        "targets_clinical_gate": {"any_deterioration": 1},
                        "phase_tag": "intervention_window",
                    }
                )
                + "\n"
            )
            protocol_path.write_text("{}\n")

            run_temporal_loop(
                str(input_path),
                str(protocol_path),
                str(out_dir),
                agent_backend="deterministic",
            )

            row = json.loads(
                (out_dir / "stage4_steward.jsonl").read_text().splitlines()[0]
            )
            metrics = json.loads((out_dir / "metrics_temporal_loop.json").read_text())
            assert_no_future_fields(row["packet"])
            self.assertEqual(row["stage4_ground_truth"]["any_deterioration"], 1)
            self.assertEqual(row["inference_context_schema"], "target_free_v1")
            self.assertTrue(metrics["target_isolation_verified"])


if __name__ == "__main__":
    unittest.main()
