import sys
import os
import unittest
import asyncio

# Ensure project root is in path
sys.path.append(os.path.dirname(__file__))

from router import classify_intent
from fusion import validate_python_code_blocks
from client import normalize_tools_for_model

class TestFable5Features(unittest.TestCase):
    
    def test_ast_validator_correct_code(self):
        # Valid python code
        text = "Here is some code:\n```python\ndef test_fn(a: int) -> int:\n    return a + 1\n```"
        errors = validate_python_code_blocks(text)
        self.assertEqual(len(errors), 0, f"Expected no syntax errors, but got: {errors}")

    def test_ast_validator_incorrect_code(self):
        # Invalid python code (missing colon after def)
        text = "Here is bad code:\n```python\ndef test_fn(a: int)\n    return a + 1\n```"
        errors = validate_python_code_blocks(text)
        self.assertEqual(len(errors), 1)
        self.assertIn("Syntax error", errors[0])
        self.assertIn("Line 1", errors[0])

    def test_tool_normalization_local_strips_tools(self):
        # Local model (mlx) should have tools stripped and instruction injected into system prompt
        kwargs = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ]
        }
        messages = [{"role": "system", "content": "Keep it short."}]
        
        new_messages, new_kwargs = normalize_tools_for_model("mlx", messages, kwargs)
        
        self.assertNotIn("tools", new_kwargs)
        self.assertNotIn("tool_choice", new_kwargs)
        self.assertTrue(len(new_messages) > 0)
        self.assertIn("TOOL CALLING SYSTEM INSTRUCTION", new_messages[0]["content"])
        self.assertIn("get_weather", new_messages[0]["content"])

    def test_tool_normalization_cloud_keeps_tools(self):
        # Cloud model (gemini or deepseek) should keep tools intact
        kwargs = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ]
        }
        messages = [{"role": "system", "content": "Keep it short."}]
        
        new_messages, new_kwargs = normalize_tools_for_model("deepseek", messages, kwargs)
        
        self.assertIn("tools", new_kwargs)
        self.assertEqual(messages, new_messages)


class TestAsyncFable5Features(unittest.IsolatedAsyncioTestCase):

    async def test_intent_classification_vision(self):
        # Message with image block should be classified as "vision"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}}
                ]
            }
        ]
        intent = await classify_intent(messages)
        self.assertEqual(intent, "vision")

    async def test_intent_classification_tools(self):
        # Message with tools parameter should be classified as "agentic_tool_calling"
        messages = [{"role": "user", "content": "Run the task"}]
        tools = [{"type": "function"}]
        
        intent = await classify_intent(messages, tools=tools)
        self.assertEqual(intent, "agentic_tool_calling")


if __name__ == "__main__":
    unittest.main()
