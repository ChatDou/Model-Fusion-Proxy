import os
import json
import logging
import time
import uuid
import asyncio
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from client import ModelAPIError, close_http_client, make_api_call, config
from router import classify_intent, check_fusion_trigger, execute_with_fallback
from fusion import execute_model_fusion

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_fusion_proxy")

app = FastAPI(title="Model Fusion Proxy Gateway")

# CORS Policy configuration with correct regex support for local dev environments
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global Concurrency Limit (Backpressure) to avoid total resource exhaustion
global_semaphore = asyncio.Semaphore(64)

# Startup hook to warm up local GPU models in the background
@app.on_event("startup")
async def startup_event():
    local_config = config.get("local", {})
    mlx_config = local_config.get("mlx", {})
    if mlx_config.get("warmup", False):
        models = mlx_config.get("models", {})
        if models:
            first_model = list(models.keys())[0]
            async def _bg_warmup():
                try:
                    logger.info(f"Triggering background local GPU warm-up for model '{first_model}'...")
                    await make_api_call(
                        first_model,
                        messages=[{"role": "user", "content": "hi"}],
                        stream=False,
                        timeout=10.0
                    )
                    logger.info("Background local GPU warm-up completed successfully.")
                except Exception as e:
                    logger.warning(f"Background local GPU warm-up failed/skipped: {e}")
            
            asyncio.create_task(_bg_warmup())

# Shutdown hook to close AsyncClient connection pooling cleanly on server exit
@app.on_event("shutdown")
async def shutdown_event():
    await close_http_client()

@app.get("/v1/models")
async def list_models():
    models_list = [
        {"id": "model-fusion", "object": "model", "owned_by": "openrouter-fusion", "permission": []},
        {"id": "openrouter/fusion", "object": "model", "owned_by": "openrouter-fusion", "permission": []},
        # ── Cloud models ─────────────────────────────────────────────────
        {"id": "deepseek-v4-pro", "object": "model", "owned_by": "deepseek", "permission": []},
        {"id": "deepseek-v4-flash", "object": "model", "owned_by": "deepseek", "permission": []},
        {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek", "permission": []},
        {"id": "glm-5.2", "object": "model", "owned_by": "zhipu", "permission": []},
        {"id": "glm-4-plus", "object": "model", "owned_by": "zhipu", "permission": []},
        {"id": "MiniMax-M3", "object": "model", "owned_by": "minimax", "permission": []},
        {"id": "abab6.5-chat", "object": "model", "owned_by": "minimax", "permission": []},
        {"id": "gemini-2.5-pro", "object": "model", "owned_by": "google", "permission": []},
        {"id": "gemini-2.5-flash", "object": "model", "owned_by": "google", "permission": []},
        {"id": "gemini-3.1-pro-preview", "object": "model", "owned_by": "google", "permission": []},
        {"id": "gemini-3.5-flash", "object": "model", "owned_by": "google", "permission": []},
        {"id": "deep-research-preview-04-2026", "object": "model", "owned_by": "google", "permission": []},
        {"id": "antigravity-preview-05-2026", "object": "model", "owned_by": "google", "permission": []},
        # ── Local models (Ollama + MLX on Apple Silicon) ─────────────────
        {"id": "llama3.2:1b", "object": "model", "owned_by": "ollama", "permission": []},
        {"id": "gemma:2b", "object": "model", "owned_by": "ollama", "permission": []},
        {"id": "gemma4-12b-it-abliterated:q4km", "object": "model", "owned_by": "ollama", "permission": []},
        {"id": "gemma4:e4b", "object": "model", "owned_by": "ollama", "permission": []},
        {"id": "Qwen3.5-9B-MLX-4bit", "object": "model", "owned_by": "mlx", "permission": []},
        {"id": "Qwen3.5-VL-9B-4bit-MLX", "object": "model", "owned_by": "mlx", "permission": []},
    ]
    return JSONResponse(content={"object": "list", "data": models_list})

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    async with global_semaphore:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        model = body.get("model", "gemini-2.5-pro")
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        tools = body.get("tools", None)
        
        passthrough_keys = ["temperature", "max_tokens", "top_p", "presence_penalty", "frequency_penalty", "stop", "response_format"]
        extra_params = {k: body[k] for k in passthrough_keys if k in body}

        if tools:
            extra_params["tools"] = tools
            if "tool_choice" in body:
                extra_params["tool_choice"] = body["tool_choice"]

        logger.info(f"Received OpenAI request for model '{model}' | Messages: {len(messages)} | Stream: {stream} | Tools: {bool(tools)}")

        category = await classify_intent(messages, tools=tools)
        logger.info(f"Intent classified category: '{category}'")

        # ────────────────────────────────────────────────────────────────────
        # P0 fix: Fusion is opt-in ONLY via explicit model name.
        #
        # History: the previous version also triggered on the substring "claude"
        # in the model name (`"claude" in model.lower()`). That looked harmless
        # but had a brutal interaction with Claude Code: Claude Code's default
        # model name is "claude-3-5-sonnet-...", which ALWAYS matches, so
        # every Claude Code request was silently routed into the 4-stage MoA
        # deliberation loop (5+ upstream LLM calls per request, 15-30s latency).
        # Hermes, by contrast, uses model names like "deepseek-v4-pro" /
        # "Qwen3.5-9B-MLX-4bit" which don't contain "claude" and so escaped
        # the trap — that's why "Hermes is fast, Claude Code is slow" was a
        # thing on this exact proxy.
        #
        # Lesson: NEVER use a client-name substring (or any other fuzzy match)
        # to switch execution paths. Magic strings = magic footguns. If you
        # want a slow-but-thorough path, make the client opt in by selecting
        # the `model-fusion` model explicitly. Otherwise the default path
        # should be the cheap, fast, fallback-chain path.
        # ────────────────────────────────────────────────────────────────────
        is_fusion_requested = model in ["model-fusion", "openrouter/fusion"]
        should_fuse = (is_fusion_requested or check_fusion_trigger(messages, category)) and not tools

        try:
            if should_fuse:
                logger.info("Executing under Model Fusion Deliberation mode")
                result = await execute_model_fusion(messages, stream=stream, category=category, **extra_params)

                if not stream:
                    # ── Non-streaming MoA fallback guard ────────────────────
                    # If fusion returns empty or near-empty content (all panels
                    # + judge failed), transparently fall back to fast path
                    # rather than returning garbage to the user.
                    content = (result.get("choices", [{}])[0].get("message", {}).get("content", "")
                               if isinstance(result, dict) else "")
                    if not content or len(str(content).strip()) < 10:
                        logger.warning(
                            "MoA fusion produced empty response (len=%d). "
                            "Falling back to fast path for category '%s'.",
                            len(str(content)) if content else 0, category
                        )
                        result = await execute_with_fallback(category, messages, stream=False, requested_model=model, **extra_params)
            else:
                logger.info(f"Executing with Fallback Chain for category: '{category}'")
                result = await execute_with_fallback(category, messages, stream=stream, requested_model=model, **extra_params)

            if stream:
                async def event_generator():
                    async for chunk in result:
                        if isinstance(chunk, bytes):
                            chunk_str = chunk.decode("utf-8")
                        else:
                            chunk_str = chunk
                        
                        # Ensure every SSE event ends with double newlines
                        if not chunk_str.endswith("\n"):
                            yield f"{chunk_str}\n\n"
                        elif chunk_str.endswith("\n") and not chunk_str.endswith("\n\n"):
                            yield f"{chunk_str}\n"
                        else:
                            yield chunk_str
                        
                return StreamingResponse(
                    event_generator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/event-stream"
                    }
                )
            else:
                return JSONResponse(content=result)

        except ModelAPIError as e:
            logger.error(f"API Error in route execution: {str(e)}")
            return JSONResponse(
                status_code=e.status_code,
                content={"error": {"message": "An error occurred executing prompt on model provider.", "type": "model_api_error", "code": e.status_code}}
            )
        except Exception as e:
            logger.error(f"Unhandled Exception in completions: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": "Internal gateway proxy error.", "type": "internal_error", "code": 500}}
            )

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """
    Anthropic-compatible Messages API.
    Translates Anthropic requests, handles text+tool_use merging, and maps SSE streaming correctly.
    """
    async with global_semaphore:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        anthropic_messages = body.get("messages", [])
        anthropic_system = body.get("system", "")
        stream = body.get("stream", False)
        model = body.get("model", "gemini-2.5-pro")
        tools = body.get("tools", None)
        
        logger.info(f"Received Anthropic request for model '{model}' | Messages: {len(anthropic_messages)} | Stream: {stream} | Tools: {bool(tools)}")

        # 1. Translate Anthropic parameters and messages to OpenAI format (combining text + tool_use blocks)
        openai_messages = []
        if anthropic_system:
            if isinstance(anthropic_system, list):
                system_text = "".join([b.get("text", "") for b in anthropic_system if b.get("type") == "text"])
            else:
                system_text = str(anthropic_system)
            openai_messages.append({"role": "system", "content": system_text})

        for msg in anthropic_messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                text_content = ""
                tool_calls = []
                tool_results = []
                
                for block in content:
                    block_type = block.get("type")
                    if block_type == "text":
                        text_content += block.get("text", "")
                    elif block_type == "tool_use":
                        tool_calls.append({
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input", {}))
                            }
                        })
                    elif block_type == "tool_result":
                        # Maps tool results back to OpenAI 'tool' role messages
                        tool_content = block.get("content")
                        if isinstance(tool_content, list):
                            tool_text = "".join([sub.get("text", "") for sub in tool_content if sub.get("type") == "text"])
                        else:
                            tool_text = str(tool_content)
                        
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id"),
                            "content": tool_text
                        })

                if role == "assistant":
                    # Merge text and tool_use in a single OpenAI assistant message
                    openai_msg = {"role": "assistant"}
                    if text_content:
                        openai_msg["content"] = text_content
                    else:
                        openai_msg["content"] = None
                    if tool_calls:
                        openai_msg["tool_calls"] = tool_calls
                    openai_messages.append(openai_msg)
                elif role == "user":
                    if tool_results:
                        openai_messages.extend(tool_results)
                    if text_content or not tool_results:
                        openai_messages.append({"role": "user", "content": text_content or ""})

        # Prepare passthrough parameters
        passthrough_keys = ["temperature", "max_tokens"]
        extra_params = {k: body[k] for k in passthrough_keys if k in body}
        
        # 2. Translates Anthropic's tools schema (input_schema) to OpenAI's tool format (parameters)
        if tools:
            openai_tools = []
            for t in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {})
                    }
                })
            extra_params["tools"] = openai_tools

        # 3. Translate Anthropic tool_choice to OpenAI tool_choice
        anthropic_tool_choice = body.get("tool_choice", None)
        if tools and anthropic_tool_choice:
            t_type = anthropic_tool_choice.get("type")
            if t_type == "auto":
                extra_params["tool_choice"] = "auto"
            elif t_type == "any":
                extra_params["tool_choice"] = "required"
            elif t_type == "tool":
                extra_params["tool_choice"] = {
                    "type": "function",
                    "function": {"name": anthropic_tool_choice.get("name")}
                }
            elif t_type == "none":
                extra_params["tool_choice"] = "none"

        # Classify and choose routing vs fusion
        category = await classify_intent(openai_messages, tools=tools)
        # P0 fix: same as OpenAI route above — only opt-in via explicit model name.
        # The Anthropic route carries the same trap: Claude Code (which speaks
        # the Anthropic `/v1/messages` API) sends model="claude-3-5-sonnet-...",
        # so any fuzzy "claude" match would force every Claude Code request into
        # 4-stage MoA. See the OpenAI handler above for the full rationale.
        is_fusion_requested = model in ["model-fusion", "openrouter/fusion"]
        should_fuse = (is_fusion_requested or check_fusion_trigger(openai_messages, category)) and not tools

        try:
            if should_fuse:
                logger.info("Anthropic route: Executing under Model Fusion Deliberation mode")
                result = await execute_model_fusion(openai_messages, stream=stream, category=category, **extra_params)

                if not stream:
                    # ── Non-streaming MoA fallback guard (same as OpenAI route) ──
                    content = (result.get("choices", [{}])[0].get("message", {}).get("content", "")
                               if isinstance(result, dict) else "")
                    if not content or len(str(content).strip()) < 10:
                        logger.warning(
                            "Anthropic route: MoA fusion produced empty response. "
                            "Falling back to fast path for category '%s'.", category
                        )
                        result = await execute_with_fallback(category, openai_messages, stream=False, requested_model=model, **extra_params)
            else:
                logger.info(f"Anthropic route: Executing with Fallback Chain for category: '{category}'")
                result = await execute_with_fallback(category, openai_messages, stream=stream, requested_model=model, **extra_params)

            msg_id = f"msg_{uuid.uuid4().hex}"

            if stream:
                async def anthropic_event_generator():
                    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
                    
                    has_started_text_block = False
                    open_tool_blocks = {} # maps tool index -> tool id
                    did_use_tools = False
                    
                    try:
                        async for chunk in result:
                            if isinstance(chunk, bytes):
                                line = chunk.decode("utf-8")
                            else:
                                line = chunk
                            
                            if not line.strip():
                                continue
                            
                            for subline in line.split("\n"):
                                subline = subline.strip()
                                if not subline or subline == "data: [DONE]":
                                    continue
                                if subline.startswith("data: "):
                                    try:
                                        data_json = json.loads(subline[6:].strip())
                                        choices = data_json.get("choices", [])
                                        if choices:
                                            delta = choices[0].get("delta", {})
                                            
                                            # Translate content block text updates
                                            content = delta.get("content", "")
                                            if content:
                                                if not has_started_text_block:
                                                    # Text block resides at index 0
                                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                                    has_started_text_block = True
                                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"
                                            
                                            # Translate tool calls block updates
                                            tool_calls = delta.get("tool_calls", None)
                                            if tool_calls:
                                                did_use_tools = True
                                                for tc in tool_calls:
                                                    tc_index = tc.get("index", 0)
                                                    # Shift tool indexes to avoid conflicts with text block (index 0)
                                                    anthropic_index = tc_index + 1
                                                    
                                                    tc_id = tc.get("id", None)
                                                    tc_func = tc.get("function", {})
                                                    tc_name = tc_func.get("name", None)
                                                    tc_args = tc_func.get("arguments", "")
                                                    
                                                    if tc_id and tc_name:
                                                        open_tool_blocks[anthropic_index] = tc_id
                                                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_index, 'content_block': {'type': 'tool_use', 'id': tc_id, 'name': tc_name, 'input': {}}})}\n\n"
                                                    if tc_args:
                                                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_index, 'delta': {'type': 'input_json_delta', 'partial_json': tc_args}})}\n\n"
                                    except Exception as ex:
                                        logger.error(f"Error parsing SSE chunk: {str(ex)}")
                                        
                    except Exception as stream_err:
                        logger.error(f"Upstream stream failed during iteration: {str(stream_err)}")
                        # Mask error and yield safely
                        if not has_started_text_block:
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                            has_started_text_block = True
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': '\n\n[Proxy Stream Error] The connection was interrupted.'}})}\n\n"
                    
                    # Yield content_block_stop for text block
                    if has_started_text_block:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                    
                    # Yield content_block_stop for all open tool blocks
                    for open_index in open_tool_blocks.keys():
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': open_index})}\n\n"
                    
                    # Verify final stop_reason dynamically based on whether tools were called
                    stop_reason = "tool_use" if did_use_tools else "end_turn"
                    
                    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
                    yield "event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n"

                return StreamingResponse(
                    anthropic_event_generator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/event-stream"
                    }
                )
            else:
                # Non-streaming translation
                choices = result.get("choices", [])
                if not choices:
                    raise ModelAPIError("Empty response from upstream", 502)
                    
                choice = choices[0]
                message_node = choice.get("message", {})
                text_out = message_node.get("content", "")
                tool_calls = message_node.get("tool_calls", None)
                
                content_blocks = []
                stop_reason = "end_turn"
                
                if text_out:
                    content_blocks.append({"type": "text", "text": text_out})
                    
                if tool_calls:
                    stop_reason = "tool_use"
                    for tc in tool_calls:
                        tc_func = tc.get("function", {})
                        try:
                            args = json.loads(tc_func.get("arguments", "{}"))
                        except Exception:
                            args = {}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id"),
                            "name": tc_func.get("name"),
                            "input": args
                        })

                anthropic_res = {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": content_blocks,
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                        "output_tokens": result.get("usage", {}).get("completion_tokens", 0)
                    }
                }
                return JSONResponse(content=anthropic_res)

        except ModelAPIError as e:
            logger.error(f"API Error in Anthropic completions: {str(e)}")
            return JSONResponse(
                status_code=e.status_code,
                content={"type": "error", "error": {"type": "api_error", "message": "Upstream service error during routing."}}
            )
        except Exception as e:
            logger.error(f"Unhandled Exception in Anthropic completions: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"type": "error", "error": {"type": "api_error", "message": "Internal gateway server error."}}
            )

@app.get("/health")
@app.get("/v1/health")
async def health_check():
    return JSONResponse(content={"status": "healthy", "timestamp": int(time.time())})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
