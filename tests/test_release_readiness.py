import hashlib
import json
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {".py", ".md", ".toml", ".txt", ".json", ".sh", ".yaml", ".yml"}
FORBIDDEN = (
    "cfnn_" + "nmi",
    "CFNN_paper_" + "nmi",
    "Nature Machine " + "Intelligence",
    "N" + "MI reproduction",
    "N" + "MI figure",
)
EXPECTED_SMOKE_SCRIPTS = (
    "figures/create_cfnn_hybrid_training_stability.py",
    "figures/create_sharp_response_mechanism.py",
    "figures/create_ai4science_fano_3d_response_recovery.py",
    "figures/create_ai4science_microstrip_recovery.py",
)


class ReleaseReadinessTest(unittest.TestCase):
    def test_apache_license_and_notice_exist(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", license_text)
        self.assertIn("Version 2.0", license_text)
        notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("Copyright 2026 CFNN Authors", notice)

    def test_project_and_import_package_are_venue_neutral(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(data["project"]["name"], "cfnn-reproduction")
        self.assertEqual(data["project"]["version"], "1.0.0")
        dependencies = data["project"].get("dependencies", [])
        for package in ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "matplotlib", "pandas"]:
            self.assertTrue(any(item.lower().startswith(package.lower()) for item in dependencies), package)
        self.assertTrue((ROOT / "src" / "cfnn" / "__init__.py").is_file())
        self.assertFalse((ROOT / "src" / ("cfnn_" + "nmi")).exists())

    def test_public_text_has_no_legacy_submission_residue(self):
        violations = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in FORBIDDEN:
                if token.lower() in text.lower():
                    violations.append(f"{path.relative_to(ROOT)}: {token}")
        self.assertEqual(violations, [])

    def test_readme_documents_caic_release_and_both_environments(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Communications AI & Computing", text)
        self.assertIn("Python 3.12.3", text)
        self.assertIn("PyTorch 2.5.1+cu124", text)
        self.assertIn("Python 3.13.5", text)
        self.assertIn("PyTorch 2.10.0+cu128", text)
        self.assertIn("minimal verification data", text.lower())
        self.assertIn("v1.0.0-caic", text)
        self.assertIn("requirements.txt", text)
        self.assertIn("requirements-lock.txt", text)

    def test_minimal_verification_manifest_matches_files(self):
        path = ROOT / "results" / "figures" / "minimal_verification_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "1.0.0")
        self.assertGreaterEqual(len(manifest["files"]), 5)
        for entry in manifest["files"]:
            target = ROOT / entry["path"]
            self.assertTrue(target.is_file(), entry["path"])
            self.assertEqual(target.stat().st_size, entry["size_bytes"])
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            self.assertEqual(digest, entry["sha256"], entry["path"])

    def test_minimal_verification_documentation_and_script_exist(self):
        doc = (ROOT / "docs" / "MINIMAL_VERIFICATION_DATA.md").read_text(encoding="utf-8")
        self.assertIn("not a retraining dataset", doc.lower())
        self.assertTrue((ROOT / "scripts" / "verify_minimal_reproduction.py").is_file())

    def test_release_notes_match_requested_release(self):
        text = (ROOT / "RELEASE_NOTES.md").read_text(encoding="utf-8")
        self.assertIn("CFNN v1.0.0", text)
        self.assertIn("Communications AI & Computing", text)
        self.assertIn("v1.0.0-caic", text)

    def test_existing_smoke_steps_are_preserved(self):
        text = (ROOT / "scripts" / "check_package.sh").read_text(encoding="utf-8")
        for script in EXPECTED_SMOKE_SCRIPTS:
            self.assertIn(script, text)
        self.assertIn("python -m compileall -q src experiments figures data", text)


if __name__ == "__main__":
    unittest.main()
