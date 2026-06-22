import asyncio
import unittest
from unittest.mock import patch, AsyncMock
import sys
import os
import json

# Ensure proxy directory is in path
sys.path.append(os.path.dirname(__file__))

from fastapi.testclient import TestClient
from main import app
import client
from client import ModelAPIError, RateLimitError, TimeoutError
import router
import fusion

class TestModelFusionProxyLogic(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.test_client = TestClient(app)

    async def test_intent_classification(self):
        """Verify that prompts are classified into appropriate categories based on keywords."""
        code_msg = [{"role": "user", "content": "How do I write a Python function that imports pandas and reads a csv file?"}]
        self.assertEqual(await router.classify_intent(code_msg), "coding")

        logic_msg = [{"role": "user", "content": "Can you solve this math equation step-by-step: 3x + 5 = 20? Prove the theorem."}]
        self.assertEqual(await router.classify_intent(logic_msg), "reasoning")

        creative_msg = [{"role": "user", "content": "Write a funny roleplay story about an AI that becomes obsessed with making paperclips."}]
        self.assertEqual(await router.classify_intent(creative_msg), "creative")

        chinese_msg = [{"role": "user", "content": "请帮我修改一下这篇政府工作报告的公文错别字并进行润色。"}]
        self.assertEqual(await router.classify_intent(chinese_msg), "chinese_nuance")

        general_msg = [{"role": "user", "content": "What is the weather like?"}]
        self.assertEqual(await router.classify_intent(general_msg), "general")

    def test_fusion_trigger(self):
        """Verify that fusion trigger identifies complex prompts or specific keywords."""
        msg1 = [{"role": "user", "content": "Compare and contrast the architectures of transformer models."}]
        self.assertTrue(router.check_fusion_trigger(msg1, "reasoning"))

        msg2 = [{"role": "user", "content": "Hello, how are you?"}]
        self.assertFalse(router.check_fusion_trigger(msg2, "general"))

        long_reasoning = "explain step by step " * 100
        msg3 = [{"role": "user", "content": f"Analyze this architecture: {long_reasoning}"}]
        self.assertTrue(router.check_fusion_trigger(msg3, "reasoning"))

    @patch("router.make_api_call", new_callable=AsyncMock)
    async def test_routing_fallback_success_on_first_try(self, mock_make_api_call):
        """Verify that non-streaming racing fallback fires all models and returns first success."""
        mock_make_api_call.return_value = {"choices": [{"message": {"content": "DeepSeek success"}}]}

        res = await router.execute_with_fallback("coding", [{"role": "user", "content": "print('hello')"}])

        self.assertEqual(res["choices"][0]["message"]["content"], "DeepSeek success")
        # Racing fallback: 1 primary + 4 fallbacks = 5 calls
        self.assertEqual(mock_make_api_call.call_count, 5)
        self.assertEqual(mock_make_api_call.call_args_list[0][0][0], "deepseek-v4-flash")

    @patch("router.make_api_call", new_callable=AsyncMock)
    async def test_routing_fallback_seamless_failover(self, mock_make_api_call):
        """Verify that when primary fails, racing fallback returns a success from one of the fallbacks."""
        # Primary fails with 429; all fallbacks succeed. Winner-takes-all will
        # pick the first fallback to resolve and cancel the rest.
        mock_make_api_call.side_effect = [
            RateLimitError("Rate limited", 429),
            {"choices": [{"message": {"content": "Gemini fallback success"}}]},
            {"choices": [{"message": {"content": "Flash success"}}]},
            {"choices": [{"message": {"content": "Pro success"}}]},
            {"choices": [{"message": {"content": "GLM success"}}]},
            {"choices": [{"message": {"content": "MiniMax success"}}]},
        ]

        res = await router.execute_with_fallback("coding", [{"role": "user", "content": "print('hello')"}])

        # Winner is one of the successful fallbacks
        self.assertIn(res["choices"][0]["message"]["content"], [
            "Gemini fallback success", "Flash success", "Pro success", "GLM success", "MiniMax success"
        ])
        # Primary must have been attempted first
        self.assertEqual(mock_make_api_call.call_args_list[0][0][0], "deepseek-v4-flash")

    @patch("fusion.make_api_call", new_callable=AsyncMock)
    async def test_fusion_deliberation_and_synthesis(self, mock_make_api_call):
        """Verify the upgraded MoA fusion pipeline."""
        mock_make_api_call.side_effect = [
            {"choices": [{"message": {"content": "Deepseek initial response"}}]},   # panel 1
            {"choices": [{"message": {"content": "GLM initial response"}}]},        # panel 2
            {"choices": [{"message": {"content": "MiniMax initial response"}}]},    # panel 3
            {"choices": [{"message": {"content": "Judge consolidated draft"}}]},    # judge draft
            {"choices": [{"message": {"content": "DeepSeek critic feedback"}}]},    # critic 1
            {"choices": [{"message": {"content": "GLM critic feedback"}}]},        # critic 2
            {"choices": [{"message": {"content": "Judge final refined answer"}}]}   # judge final
        ]
        
        with patch.dict(fusion.config["fusion"], {"strategy": "cloud"}):
            res = await fusion.execute_model_fusion([{"role": "user", "content": "Compare A and B"}], stream=False)
            self.assertEqual(res["choices"][0]["message"]["content"], "Judge final refined answer")
            self.assertEqual(mock_make_api_call.call_count, 7)

    @patch("main.execute_with_fallback", new_callable=AsyncMock)
    def test_anthropic_messages_non_streaming_translation(self, mock_execute_fallback):
        """Verify translation of Anthropic payload (text + tools) to OpenAI and back for non-streaming calls."""
        # Setup mock OpenAI output containing text & tool_calls
        mock_execute_fallback.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I will run the tool now.",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\": \"/src/main.py\"}"
                        }
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20
            }
        }

        payload = {
            "model": "claude-latest",
            "messages": [
                {"role": "user", "content": "Please inspect main.py."}
            ],
            "system": "System instructions.",
            "tools": [{
                "name": "read_file",
                "description": "Read file contents",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}}
                }
            }],
            "tool_choice": {"type": "tool", "name": "read_file"},
            "stream": False
        }

        response = self.test_client.post("/v1/messages", json=payload)
        self.assertEqual(response.status_code, 200)
        
        res_json = response.json()
        self.assertEqual(res_json["stop_reason"], "tool_use")
        self.assertEqual(res_json["content"][0]["type"], "text")
        self.assertEqual(res_json["content"][1]["type"], "tool_use")
        self.assertEqual(res_json["content"][1]["id"], "call_123")
        self.assertEqual(res_json["content"][1]["name"], "read_file")
        self.assertEqual(res_json["content"][1]["input"]["path"], "/src/main.py")
        self.assertEqual(res_json["usage"]["input_tokens"], 10)
        self.assertEqual(res_json["usage"]["output_tokens"], 20)

        # Check request translation
        called_args = mock_execute_fallback.call_args[0]
        called_messages = called_args[1]
        
        # System message is prepended
        self.assertEqual(called_messages[0]["role"], "system")
        self.assertEqual(called_messages[0]["content"], "System instructions.")
        
        # Tools parsed into extra params
        called_kwargs = mock_execute_fallback.call_args[1]
        self.assertIn("tools", called_kwargs)
        self.assertEqual(called_kwargs["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(called_kwargs["tool_choice"]["function"]["name"], "read_file")

    @patch("main.execute_with_fallback", new_callable=AsyncMock)
    def test_anthropic_messages_streaming_translation(self, mock_execute_fallback):
        """Verify translation of OpenAI SSE chunk updates into correct Anthropic content/tool events."""
        async def mock_openai_stream():
            # SSE chunk 1: text delta
            yield b'data: {"choices": [{"delta": {"content": "Streaming text."}}]}\n'
            # SSE chunk 2: tool call delta start
            yield b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_abc", "function": {"name": "run_command"}}]}}]}\n'
            # SSE chunk 3: tool call delta arguments
            yield b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"cmd\\": \\"ls\\"}"}}]}}]}\n'
            # SSE chunk 4: end
            yield b'data: [DONE]\n'

        mock_execute_fallback.return_value = mock_openai_stream()

        payload = {
            "model": "claude-latest",
            "messages": [{"role": "user", "content": "Run command."}],
            "tools": [{
                "name": "run_command",
                "description": "Run command description",
                "input_schema": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}}
                }
            }],
            "stream": True
        }

        response = self.test_client.post("/v1/messages", json=payload)
        self.assertEqual(response.status_code, 200)

        lines = []
        for line in response.iter_lines():
            if line:
                if isinstance(line, bytes):
                    lines.append(line.decode("utf-8"))
                else:
                    lines.append(line)
        
        # We search for the occurrence of standard Anthropic stream events
        has_message_start = False
        has_text_block_start = False
        has_text_block_delta = False
        has_tool_block_start = False
        has_tool_block_delta = False
        has_content_stop_text = False
        has_content_stop_tool = False
        has_message_delta = False
        
        stop_reason = None
        
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:].strip()
                if event_type == "message_start":
                    has_message_start = True
                elif event_type == "content_block_start":
                    pass
                elif event_type == "content_block_delta":
                    pass
                elif event_type == "content_block_stop":
                    pass
                elif event_type == "message_delta":
                    has_message_delta = True
                    
            elif line.startswith("data: "):
                data = json.loads(line[6:].strip())
                data_type = data.get("type")
                if data_type == "content_block_start":
                    block = data.get("content_block", {})
                    if block.get("type") == "text" and data.get("index") == 0:
                        has_text_block_start = True
                    elif block.get("type") == "tool_use" and data.get("index") == 1:
                        # Shifted block index tc_index + 1 = 1
                        has_tool_block_start = True
                elif data_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta" and data.get("index") == 0:
                        has_text_block_delta = True
                    elif delta.get("type") == "input_json_delta" and data.get("index") == 1:
                        has_tool_block_delta = True
                elif data_type == "content_block_stop":
                    if data.get("index") == 0:
                        has_content_stop_text = True
                    elif data.get("index") == 1:
                        has_content_stop_tool = True
                elif data_type == "message_delta":
                    stop_reason = data.get("delta", {}).get("stop_reason")

        self.assertTrue(has_message_start)
        self.assertTrue(has_text_block_start)
        self.assertTrue(has_text_block_delta)
        self.assertTrue(has_tool_block_start)
        self.assertTrue(has_tool_block_delta)
        self.assertTrue(has_content_stop_text)
        self.assertTrue(has_content_stop_tool)
        self.assertTrue(has_message_delta)
        self.assertEqual(stop_reason, "tool_use")  # Verified stop reason set to tool_use

if __name__ == "__main__":
    unittest.main()
