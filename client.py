import os
import re
import json
import logging
import asyncio
import yaml
import httpx
from typing import AsyncGenerator, Dict, Any, List, Optional

logger = logging.getLogger("model_fusion_proxy.client")

# Load Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Exceptions
class ModelAPIError(Exception):
    """Base exception for model API errors."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code

class RateLimitError(ModelAPIError):
    """Raised when HTTP 429 is received."""
    pass

class TimeoutError(ModelAPIError):
    """Raised when request times out."""
    pass

# ─── Load .env once at module init ────────────────────────────────────────
# Previously get_api_key() opened and parsed the .env file on every single API
# call, which under high concurrency (64 parallel requests, each hitting 3+
# upstream models for a 4-stage MoA loop) created pointless disk I/O and file
# descriptor churn.  Cache it once at import time.
_ENV_CACHE: Dict[str, str] = {}
_env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_file):
    with open(_env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                _ENV_CACHE[k.strip()] = v.strip()

# Global Singleton HTTPX Client to manage connection pooling and prevent resource leakages
_client_limits = httpx.Limits(max_keepalive_connections=40, max_connections=100, keepalive_expiry=60.0)
http_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()

# Concurrency lock for local GPU models (Ollama/MLX) to prevent resource contention/OOM
_policy = config.get("local_model_policy", {})
_max_local_concurrent = _policy.get("max_concurrent", 2)
local_gpu_semaphore = asyncio.Semaphore(_max_local_concurrent)


async def get_http_client() -> httpx.AsyncClient:
    global http_client
    if http_client is None or http_client.is_closed:
        async with _client_lock:
            # Double-check inside the lock to avoid a TOCTOU race:
            # two coroutines can both pass the outer `is None` check,
            # but only the first one through the lock creates the client.
            if http_client is None or http_client.is_closed:
                http_client = httpx.AsyncClient(limits=_client_limits)
    return http_client

async def close_http_client():
    global http_client
    if http_client is not None and not http_client.is_closed:
        await http_client.aclose()
        logger.info("Global HTTPX AsyncClient closed.")

# Helper to resolve API keys from environment
# Uses the module-level _ENV_CACHE populated once at import time, so
# high-concurrency paths don't re-read the .env file on every call.
def get_api_key(provider: str) -> str:
    env_var_name = f"{provider.upper()}_API_KEY"
    key = os.environ.get(env_var_name) or _ENV_CACHE.get(env_var_name)

    if not key:
        key_tmpl = config.get("providers", {}).get(provider, {}).get("api_key", "")
        if key_tmpl.startswith("${") and key_tmpl.endswith("}"):
            var_name = key_tmpl[2:-1]
            key = os.environ.get(var_name, "") or _ENV_CACHE.get(var_name, "")
        else:
            key = key_tmpl
    return key

# Resolve model name to provider config.
# Priority: local (ollama / mlx) first → cloud providers → name-based fallback.
def resolve_model(model_name: str) -> Dict[str, Any]:
    # ── Phase 1: Check local providers (ollama, mlx) ──────────────────
    local_config = config.get("local", {})
    for local_name, local_data in local_config.items():
        models_map = local_data.get("models", {})
        for config_key, real_model_id in models_map.items():
            if real_model_id == model_name or config_key == model_name:
                return {
                    "provider": local_name,           # "ollama" or "mlx"
                    "real_model": real_model_id,
                    "base_url": local_data.get("base_url"),
                    "api_key": local_data.get("api_key", ""),
                    "is_local": True,                  # Signal: no retry, short timeout
                }

    # ── Phase 2: Check cloud providers ────────────────────────────────
    providers = config.get("providers", {})
    for provider_name, provider_data in providers.items():
        models_map = provider_data.get("models", {})
        for config_key, real_model_id in models_map.items():
            if real_model_id == model_name or config_key == model_name:
                return {
                    "provider": provider_name,
                    "real_model": real_model_id,
                    "base_url": provider_data.get("base_url"),
                    "api_key": get_api_key(provider_name),
                    "is_local": False,
                }

    # ── Phase 3: Name-based heuristic fallback ─────────────────────────
    if "gemini" in model_name:
        return {"provider": "gemini", "real_model": model_name, "base_url": providers["gemini"]["base_url"], "api_key": get_api_key("gemini"), "is_local": False}
    if "deepseek" in model_name:
        return {"provider": "deepseek", "real_model": model_name, "base_url": providers["deepseek"]["base_url"], "api_key": get_api_key("deepseek"), "is_local": False}
    if "glm" in model_name or "chatglm" in model_name:
        return {"provider": "glm", "real_model": model_name, "base_url": providers["glm"]["base_url"], "api_key": get_api_key("glm"), "is_local": False}
    if "abab" in model_name or "minimax" in model_name:
        return {"provider": "minimax", "real_model": model_name, "base_url": providers["minimax"]["base_url"], "api_key": get_api_key("minimax"), "is_local": False}

    raise ModelAPIError(f"Unknown model name: {model_name}", 400)

def normalize_tools_for_model(provider: str, messages: List[Dict[str, str]], kwargs: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Normalizes tools and tool_choice across different providers.
    - For local providers (ollama/mlx), strips native tools/tool_choice to avoid API crashes,
      and appends tool instructions to the system prompt.
    - For cloud providers, ensures the tool format is compatible.
    """
    tools = kwargs.get("tools")
    if not tools:
        return messages, kwargs

    new_kwargs = dict(kwargs)
    
    if provider in ["mlx", "ollama"]:
        new_kwargs.pop("tools", None)
        new_kwargs.pop("tool_choice", None)
        
        tool_desc = []
        for t in tools:
            if t.get("type") == "function":
                f = t.get("function", {})
                tool_desc.append(
                    f"Name: {f.get('name')}\n"
                    f"Description: {f.get('description', '')}\n"
                    f"Parameters JSON Schema: {json.dumps(f.get('parameters', {}))}\n"
                )
        
        tool_instruction = (
            "\n[TOOL CALLING SYSTEM INSTRUCTION]\n"
            "You have access to the following tools. If you need to call a tool, "
            "respond ONLY with a JSON object in this format:\n"
            "```json\n"
            "{\n"
            "  \"tool_calls\": [\n"
            "    {\n"
            "      \"id\": \"call_unique_id\",\n"
            "      \"type\": \"function\",\n"
            "      \"function\": {\n"
            "        \"name\": \"tool_name\",\n"
            "        \"arguments\": \"{\\\"param1\\\": \\\"value\\\"}\"\n"
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            "Available Tools:\n" + "\n".join(tool_desc) + "\n"
            "If no tool needs to be called, answer normally."
        )
        
        new_messages = []
        injected = False
        for msg in messages:
            if msg.get("role") == "system":
                new_messages.append({
                    "role": "system",
                    "content": msg.get("content", "") + tool_instruction
                })
                injected = True
            else:
                new_messages.append(msg)
        
        if not injected:
            new_messages.insert(0, {
                "role": "system",
                "content": tool_instruction
            })
        
        return new_messages, new_kwargs

    return messages, new_kwargs

async def make_api_call(
    model_name: str,
    messages: List[Dict[str, str]],
    stream: bool = False,
    **kwargs
) -> Any:
    """
    Makes an asynchronous HTTP request using the global connection pool.
    Local models (ollama/mlx) get shorter timeouts and no retries — OOM
    or warm-up failures won't be fixed by hammering the local GPU.
    """
    info = resolve_model(model_name)
    provider = info["provider"]
    real_model = info["real_model"]
    base_url = info["base_url"]
    api_key = info["api_key"]
    is_local = info.get("is_local", False)

    headers = {
        "Content-Type": "application/json",
    }
    # Local models (ollama/mlx) never need auth headers — their API keys
    # are dummy placeholders in config.yaml.  Sending a Bearer token to
    # mlx_lm.server will trigger a 401.  Cloud providers always need auth.
    if not is_local and api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif not is_local and not api_key:
        raise ModelAPIError(f"API Key for provider '{provider}' is not set. Please set the {provider.upper()}_API_KEY environment variable.", 401)
    if provider == "gemini":
        headers["Connection"] = "close"

    retry_settings = config.get("retry", {})
    # Local models: single attempt, short timeout.  GPU OOM or warm-up
    # latency aren't transient network errors — retrying won't help.
    max_attempts = 1 if is_local else kwargs.pop("max_attempts", retry_settings.get("max_attempts", 3))
    backoff_factor = retry_settings.get("backoff_factor", 1.5)
    default_timeout = 10 if is_local else retry_settings.get("timeout_seconds", 30)

    timeout_val = kwargs.pop("timeout", default_timeout)
    client_timeout = httpx.Timeout(timeout_val, connect=5.0 if is_local else 10.0)

    norm_messages, norm_kwargs = normalize_tools_for_model(provider, messages, kwargs)

    payload = {
        "model": real_model,
        "messages": norm_messages,
        "stream": stream,
        **norm_kwargs
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    client = await get_http_client()

    # Semaphore lock management for local GPU models (Ollama/MLX)
    if is_local:
        logger.info(f"Acquiring local GPU semaphore for {real_model}...")
        await local_gpu_semaphore.acquire()

    semaphore_released = False
    def release_sem():
        nonlocal semaphore_released
        if is_local and not semaphore_released:
            local_gpu_semaphore.release()
            semaphore_released = True
            logger.info(f"Released local GPU semaphore for {real_model}.")

    try:
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            response = None
            try:
                logger.info(f"Sending request to {provider} ({real_model}), attempt {attempt}/{max_attempts}")
                
                if not stream:
                    response = await client.post(url, headers=headers, json=payload, timeout=client_timeout)
                    if response.status_code == 200:
                        release_sem()
                        return response.json()
                    elif response.status_code == 429:
                        raise RateLimitError("Rate limit exceeded by upstream provider", 429)
                    else:
                        err_body = response.text
                        logger.error(f"Upstream provider error {response.status_code}: {err_body}")
                        raise ModelAPIError(f"Upstream provider error: {response.status_code} - {err_body}", response.status_code)
                else:
                    # Streaming response
                    req = client.build_request("POST", url, headers=headers, json=payload, timeout=client_timeout)
                    response = await client.send(req, stream=True)
                    
                    if response.status_code != 200:
                        try:
                            body = await response.aread()
                            body_text = body.decode("utf-8", errors="ignore")
                        finally:
                            await response.aclose()
                        
                        if response.status_code == 429:
                            raise RateLimitError("Rate limit exceeded by upstream provider (stream)", 429)
                        else:
                            logger.error(f"Upstream provider error (stream) {response.status_code}: {body_text}")
                            raise ModelAPIError(f"Upstream provider error (stream): {response.status_code} - {body_text}", response.status_code)
                    
                    async def response_stream_generator() -> AsyncGenerator[str, None]:
                        try:
                            async for line in response.aiter_lines():
                                if line:
                                    yield line
                        finally:
                            # Guarantee response cleanup on exit/close
                            try:
                                await response.aclose()
                            finally:
                                release_sem()
                    
                    return response_stream_generator()

            except (httpx.TimeoutException, httpx.ConnectTimeout) as e:
                logger.warning(f"Timeout on {provider} ({real_model}), attempt {attempt}")
                if attempt == max_attempts:
                    raise TimeoutError(f"Request to {provider} timed out after {timeout_val} seconds.", 504)
            except (httpx.NetworkError, httpx.ProtocolError) as e:
                logger.warning(f"Network error on {provider} ({real_model}), attempt {attempt}: {str(e)}")
                if attempt == max_attempts:
                    raise ModelAPIError(f"Network error contacting {provider}.", 502)
            except ModelAPIError as e:
                if e.status_code in [400, 401, 403]:
                    raise e
                if attempt == max_attempts:
                    raise e
            
            # Exponential backoff
            sleep_time = backoff_factor ** attempt
            await asyncio.sleep(sleep_time)

        raise ModelAPIError("Failed after maximum retries", 500)
    except Exception as e:
        release_sem()
        raise e
