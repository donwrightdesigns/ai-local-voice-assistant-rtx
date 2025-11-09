import unittest
from src.assistant import Assistant

class TestProviderSelection(unittest.TestCase):
    def test_default_provider_init(self):
        a = Assistant(provider="ollama")
        self.assertEqual(a.state, "idle")

if __name__ == "__main__":
    unittest.main()
