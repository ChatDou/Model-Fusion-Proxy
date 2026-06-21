import asyncio
import json
import logging
import yaml
import os
import re
from typing import List, Dict, Any, Tuple
from client import make_api_call, ModelAPIError, resolve_model

logger = logging.getLogger("model_fusion_proxy.router")

# Load Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Word boundaries pattern compilation to avoid partial substring matching issues (e.g. "js" in "adjust")
CODING_KEYWORDS_RE = re.compile(
    r"\b(def|class|import|function|const|let|void|int|public|private|static|python|javascript|golang|rust|sql|html|css|docker|kubernetes|k8s|git|api|json|xml|yaml|code|bug|exception|compile|error|regex|debug|leetcode|github|database|postgresql|mysql|sqlite|nosql|mongodb|redis|query|table|select|schema)\b|(\bjs\b|\bts\b)",
    re.IGNORECASE
)

REASONING_KEYWORDS_RE = re.compile(
    r"\b(reason|logic|solve|math|proof|theorem|calculate|equation|derivation|step-by-step|why|consequence|diagnose|architecture|trade-off|feasibility|analysis|evaluate|design|system|distributed|gateway|consistency|clock|compare|contrast|scalability|latency)\b",
    re.IGNORECASE
)

CREATIVE_KEYWORDS_RE = re.compile(
    r"\b(story|poem|novel|roleplay|mimic|creative|joke|fiction|scenario|brainstorm|copywriting|slogan|marketing|ads)\b|"
    r"(故事|小说|诗歌|角色扮演|拟人|创意|脑洞|文案|广告词)",
    re.IGNORECASE
)

CHINESE_NUANCE_KEYWORDS_RE = re.compile(
    r"(公文|体制|汇报|本地化|翻译|译文|白话|古文|古诗|文言文|成语|歇后语|传统文化|政策解读|申论|报告|修辞|赏析|背景|历史|启示|魏征|太宗|新质|供给侧|唐朝|贞观|共同富裕)"
)

async def classify_intent(messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> str:
    """
    Two-stage intent classifier.

    Stage 1 (O(1) regex): nanosecond keyword matching for obvious coding,
    reasoning, creative, and Chinese-nuance tasks.  Handles ~70% of traffic
    with zero latency and zero API cost.

    Stage 2 (local LLM): when Stage 1 returns all-zero (regex couldn't
    classify), call a local MLX model (Qwen3.5-9B) with strict JSON output
    to do semantic classification.  Adds ~200-500ms but catches cases
    where keywords are absent yet intent is clear (e.g. "帮我优化这个脚本"
    has no coding keywords but IS a coding task).

    Stage 2 failures (timeout, model not running) silently degrade to
    "general" — the fallback chain for general is long enough to cover
    most tasks.
    """
    if tools:
        logger.info("Tools detected in request. Routing directly to agentic_tool_calling pipeline.")
        return "agentic_tool_calling"

    # Detect if multimodal image content exists in the prompt
    has_image = False
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image_url":
                        has_image = True
                        break
            if has_image:
                break
    if has_image:
        logger.info("Image content detected in request. Routing directly to vision pipeline.")
        return "vision"

    # Safely extract text contents from user prompts (supporting multi-turn and complex content lists)
    user_prompts = []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                text_content = ""
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content += block.get("text", "")
                user_prompts.append(text_content)
            elif isinstance(content, str):
                user_prompts.append(content)

    full_prompt = "\n".join(user_prompts)
    if not full_prompt.strip():
        return "general"

    # ── Stage 1: O(1) regex fast path ─────────────────────────────────
    coding_count = len(CODING_KEYWORDS_RE.findall(full_prompt))
    reasoning_count = len(REASONING_KEYWORDS_RE.findall(full_prompt))
    creative_count = len(CREATIVE_KEYWORDS_RE.findall(full_prompt))
    chinese_count = len(CHINESE_NUANCE_KEYWORDS_RE.findall(full_prompt))

    logger.debug(f"Regex classification scores -> coding: {coding_count}, reasoning: {reasoning_count}, creative: {creative_count}, chinese: {chinese_count}")

    max_score = max(coding_count, reasoning_count, creative_count, chinese_count)
    if max_score > 0:
        if max_score == coding_count:
            return "coding"
        elif max_score == reasoning_count:
            return "reasoning"
        elif max_score == creative_count:
            return "creative"
        else:
            return "chinese_nuance"

    # ── Stage 2: Local LLM semantic classifier ────────────────────────
    classifier_config = config.get("routing", {}).get("classifier", {})
    if not classifier_config.get("enabled", False):
        return "general"

    stage2_model = classifier_config.get("stage2_model", "qwen35_9b")
    stage2_timeout = classifier_config.get("stage2_timeout", 2.0)
    fallback_intent = classifier_config.get("stage2_fallback_intent", "general")

    system_prompt = (
        "You are a request classifier for an LLM routing proxy. "
        "Analyze the user's message and output exactly ONE category in JSON: "
        '{"intent": "coding|reasoning|creative|chinese_nuance|general"}. '
        "Rules:\n"
        "- coding: programming, debugging, algorithms, SQL, math proofs, JSON/YAML output\n"
        "- reasoning: logic puzzles, step-by-step analysis, architecture design, trade-off evaluation\n"
        "- creative: story writing, poetry, copywriting, roleplay, brainstorming, slogans\n"
        "- chinese_nuance: Chinese document polishing, classical poetry, policy interpretation, local context\n"
        "- general: casual chat, simple questions, greetings, weather, facts\n"
        "Output ONLY the JSON object, no other text."
    )

    try:
        response = await asyncio.wait_for(
            make_api_call(
                stage2_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt[:2000]}
                ],
                stream=False,
                timeout=stage2_timeout,
            ),
            timeout=stage2_timeout
        )
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Parse JSON — model may wrap in markdown code blocks
        content_clean = content.strip()
        if content_clean.startswith("```"):
            lines = content_clean.split("\n")
            content_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else lines[-1]
            content_clean = content_clean.replace("```json", "").replace("```", "").strip()
        result = json.loads(content_clean)
        intent = result.get("intent", fallback_intent)
        if intent in ("coding", "reasoning", "creative", "chinese_nuance", "general"):
            logger.info(f"Stage-2 local classifier → {intent} (prompt: {full_prompt[:80]}...)")
            return intent
        else:
            logger.warning(f"Stage-2 classifier returned unknown intent '{intent}', falling back to '{fallback_intent}'")
            return fallback_intent

    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Stage-2 local classifier failed ({type(e).__name__}: {str(e)[:100]}). "
                       f"Falling back to '{fallback_intent}'.")
        return fallback_intent

def check_fusion_trigger(messages: List[Dict[str, str]], category: str) -> bool:
    """
    Check if the request should trigger Model Fusion mode.
    """
    fusion_config = config.get("fusion", {})
    if not fusion_config.get("enabled", False):
        return False

    user_prompts = []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                user_prompts.append(content)
            elif isinstance(content, list):
                text_content = "".join([b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"])
                user_prompts.append(text_content)

    full_prompt = "\n".join(user_prompts)
    full_prompt_lower = full_prompt.lower()

    # Keyword match
    for kw in fusion_config.get("keywords", []):
        if kw in full_prompt_lower:
            logger.info(f"Fusion triggered by keyword: '{kw}'")
            return True

    # Complexity match (Reasoning + long prompt > 1000 characters)
    if category == "reasoning" and len(full_prompt) > 1000:
        logger.info("Fusion triggered by reasoning complexity (prompt length > 1000).")
        return True

    return False

def get_routing_chain(category: str) -> Tuple[str, List[str]]:
    cat_config = config.get("routing", {}).get("categories", {}).get(category, {})
    primary = cat_config.get("primary")
    fallbacks = cat_config.get("fallbacks", [])
    return primary, fallbacks

async def execute_with_fallback(
    category: str,
    messages: List[Dict[str, str]],
    stream: bool = False,
    requested_model: str = None,
    **kwargs
) -> Any:
    """
    Executes a model completion call using the primary model and its fallback list.

    ─────────────────────────────────────────────────────────────────────
    DESIGN NOTE (read this before changing the logic below)
    ─────────────────────────────────────────────────────────────────────
    Two execution modes that look similar but have very different semantics:

    1. NON-STREAMING (batch) — we can race all models concurrently and return the
       first success. If the primary is rate-limited (429) or temporarily down (503),
       we don't want the user to wait 5+ seconds sequentially walking the fallback
       list. The non-streaming response is atomic: the client gets the full JSON
       blob only after we decide which model won, so it's safe to cancel the losers.

       Why `asyncio.wait` + `FIRST_COMPLETED` instead of `asyncio.as_completed`:
       In Python 3.13+ `asyncio.as_completed()` yields *coroutine objects* from its
       internal `_wait_for_one` method, NOT task objects. If you `await` one and then
       try to look it up in a `task → name` dict, you get a KeyError. Use
       `asyncio.wait()` which gives you the actual `asyncio.Task` objects, on which
       `.result()` / `.exception()` / `.cancel()` work normally.

    2. STREAMING — once we've yielded the first SSE chunk to the client, we CANNOT
       switch to a different model mid-stream. The HTTP response has already started
       and the client has already begun rendering. So we fall back sequentially:
       try primary, if it fails before yielding the first token, try fallback #1,
       etc. Because the primary in this codebase is the lowest-latency tier
       (deepseek-v4-flash, deepseek-v4-pro), streaming latency is usually sub-second
       and the sequential fallback rarely matters in practice.

    Cancellation discipline (the part most code gets wrong):
    - When the winner is found, we MUST cancel the losers; otherwise they keep
      burning tokens until their natural completion.
    - On any exception path, the `finally` block guarantees no task is leaked.
    - Cancellation propagates through `httpx.AsyncClient` requests: the in-flight
      HTTP request gets an `asyncio.CancelledError`, the connection pool is freed.
    ─────────────────────────────────────────────────────────────────────
    """
    primary, fallbacks = get_routing_chain(category)
    if requested_model and requested_model not in ["model-fusion", "openrouter/fusion"]:
        primary = requested_model
        fallbacks = [f for f in fallbacks if f != requested_model]
    model_queue = [primary] + fallbacks

    if not stream:
        # ─── Non-streaming racing fallback ────────────────────────────────
        # Fire all candidate models in parallel. Whichever returns first (or
        # succeeds first if multiple succeed near-simultaneously) wins; the rest
        # are cancelled to stop them from consuming tokens.
        async def _safe_call(m: str):
            """Wrap make_api_call so a single failure doesn't kill the race."""
            try:
                return ("ok", await make_api_call(m, messages, stream=False, **kwargs))
            except ModelAPIError as e:
                # Expected: rate limit / 5xx from upstream. Treat as a normal
                # "this candidate lost" signal, not a crash.
                return ("err", e)
            except Exception as e:
                # ProgrammingError / ConnectionError / etc. Log but don't kill
                # the whole race — let other candidates try.
                return ("err", e)

        # Schedule every model in the fallback chain as an independent task.
        task_to_model = {asyncio.create_task(_safe_call(m)): m for m in model_queue}

        try:
            # Loop until we have a winner or all candidates have been tried.
            while task_to_model:
                # Wait for at least one task to finish. The `timeout` here is an
                # *overall* budget for the whole race, not per-model.
                done, pending = await asyncio.wait(
                    task_to_model.keys(),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=kwargs.get("timeout", 30)
                )

                if not done:
                    # Race timed out entirely — no model finished in budget.
                    # Cancel everything and bail out (the except below raises 500).
                    for t in pending:
                        t.cancel()
                    break

                for task in done:
                    try:
                        status, payload = task.result()
                    except Exception as e:
                        # Defensive: _safe_call should never raise, but if a
                        # task itself was cancelled mid-await, treat as "lost".
                        status, payload = "err", e

                    if status == "ok":
                        # WINNER. Immediately cancel the rest so they stop
                        # burning tokens. This is the part that turns "user
                        # waits 10s for slow primary" into "user gets response
                        # in 1.2s, slower models quietly cancelled".
                        for t in pending:
                            t.cancel()
                        logger.info(f"Non-streaming fallback winner: {task_to_model[task]}")
                        return payload
                    else:
                        # This candidate failed. Log it, drop it from the map,
                        # and continue waiting for the others to finish.
                        logger.warning(f"Model {task_to_model[task]} failed: {str(payload)}")

                # Garbage-collect finished (failed) tasks so the next iteration
                # of the while loop only waits on the survivors.
                for task in done:
                    task_to_model.pop(task, None)
        finally:
            # Belt-and-suspenders: if we exit the function via exception or
            # `break`, make sure no task is left running. asyncio will warn
            # loudly about unawaited tasks otherwise ("Task was destroyed but
            # it is pending").
            for t in list(task_to_model.keys()):
                if not t.done():
                    t.cancel()
            if task_to_model:
                # Wait for cancellations to actually propagate (the await gives
                # CancelledError a chance to reach the inner coroutine and
                # release httpx connections).
                await asyncio.gather(*task_to_model.keys(), return_exceptions=True)

        logger.critical("All models in the routing/fallback queue failed.")
        raise ModelAPIError(
            "All routed models failed.",
            status_code=500
        )

    # ─── Streaming sequential fallback ───────────────────────────────────
    # No race here: once we yield the first SSE chunk, the HTTP response has
    # already started and we cannot retroactively switch models. So we walk
    # the fallback chain in order, and only "fall back" if the chosen model
    # fails *before* the first chunk reaches the client.
    last_error = None
    for idx, model_name in enumerate(model_queue):
        try:
            logger.info(f"Attempting streaming execution using model: {model_name} (Position {idx + 1}/{len(model_queue)} in fallback chain)")
            result = await make_api_call(model_name, messages, stream=stream, **kwargs)
            return result
        except ModelAPIError as e:
            last_error = e
            logger.error(f"Model {model_name} failed with status {e.status_code}: {str(e)}")
            continue
        except Exception as e:
            # Same policy as non-streaming: log + try next, don't crash the request.
            logger.error(f"System/Network exception during execution using {model_name}: {str(e)}")
            last_error = e
            continue

    logger.critical("All models in the routing/fallback queue failed.")
    raise ModelAPIError(
        f"All routed models failed. Last error: {str(last_error)}",
        status_code=getattr(last_error, "status_code", 500)
    )
