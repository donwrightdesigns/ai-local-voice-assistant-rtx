import unittest
from pathlib import Path
from src.config_loader import load_config, apply_computed_defaults

class TestConfigLoader(unittest.TestCase):
    def test_merge_and_defaults(self):
        # Use repo root relative resolution
        project_root = Path(__file__).parents[2]
        cfg, _ = load_config(project_root)
        merged = apply_computed_defaults(cfg)
        # Keys exist
        self.assertIn("stt", merged)
        self.assertIn("llm", merged)
        self.assertIn("wake_mode", merged)
        # Profiles resolved presence
        self.assertIn("profiles", merged["stt"]) 

if __name__ == "__main__":
    unittest.main()
