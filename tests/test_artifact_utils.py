import json
import tempfile
import unittest
import warnings
from pathlib import Path

from mxmoe.quant.artifact_utils import (
    load_optional_metadata,
    metadata_path,
    validate_calibration_artifacts,
    write_metadata,
)


class ArtifactUtilsTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "dataset": "c4",
            "nsamples": 128,
            "seqlen": 2048,
            "seed": 0,
        }

    def metadata(self, input_hash="abc"):
        return {
            "calibration": {
                **self.expected,
                "input_ids_sha256": input_hash,
            }
        }

    def test_matching_artifacts_are_accepted(self):
        validate_calibration_artifacts(
            [("trace", self.metadata()), ("loss", self.metadata())],
            self.expected,
        )

    def test_field_mismatch_is_rejected(self):
        metadata = self.metadata()
        metadata["calibration"]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_calibration_artifacts([("loss", metadata)], self.expected)

    def test_hash_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "token hashes"):
            validate_calibration_artifacts(
                [("trace", self.metadata("a")), ("loss", self.metadata("b"))],
                self.expected,
            )

    def test_legacy_artifact_without_sidecar_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "legacy.json"
            artifact.write_text("{}", encoding="utf-8")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertIsNone(
                    load_optional_metadata(artifact, artifact_name="legacy")
                )
            self.assertIn("legacy artifact", str(caught[0].message))

    def test_sidecar_round_trip_keeps_artifact_name(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "allocation.json"
            artifact.write_text("{}", encoding="utf-8")
            sidecar = write_metadata(artifact, self.metadata())
            self.assertEqual(sidecar, metadata_path(artifact))
            loaded = load_optional_metadata(artifact, artifact_name="allocation")
            self.assertEqual(loaded["calibration"], self.metadata()["calibration"])
            self.assertIn("artifact_sha256", loaded)

    def test_sidecar_detects_changed_artifact_content(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "allocation.json"
            artifact.write_text("{}", encoding="utf-8")
            write_metadata(artifact, self.metadata())
            artifact.write_text(json.dumps({"changed": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content does not match"):
                load_optional_metadata(artifact, artifact_name="allocation")


if __name__ == "__main__":
    unittest.main()
