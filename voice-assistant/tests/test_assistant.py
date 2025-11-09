import unittest
from src.assistant import Assistant

class TestAssistant(unittest.IsolatedAsyncioTestCase):
    async def test_state_transitions(self):
        # Basic sanity check – we can instantiate and run the state machine
        a = Assistant(provider="openai")
        self.assertEqual(a.state, "idle")
        # We don't actually record audio in the test; just simulate a query
        await a._process_query("Hello world")
        self.assertEqual(a.state, "idle")   # back to idle after processing

if __name__ == "__main__":
    unittest.main()
