"""
Shared LLM client helper.
 
Centralizes the OpenAI-compatible client construction so every agent
(profiler, sttm_generator, bronze_agent, silver_agent, gold_agent,
reporter) calls the same _get_client() / call_llm() pair instead of
duplicating the provider wiring six times. If you ever swap providers
again, this is the only file that needs to change.
 
Currently wired for Groq (OpenAI-compatible endpoint).
"""
import re
import time
 
from openai import OpenAI, RateLimitError
 
from core.config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL
 
_client = None
 
 
def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        _client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)
    return _client
 
 
def _seconds_to_wait(error: RateLimitError) -> float:
    """
    Groq's 429 response includes a Retry-After-style hint both as a header
    and in the error message text ("Please try again in 14.9625s"). A full
    pipeline run makes 6+ calls in quick succession, so hitting the
    account's rolling per-minute token budget partway through a run is
    expected, not a bug -- the fix is to wait exactly as long as Groq says
    to, then retry, rather than fail the whole run.
    """
    try:
        header_val = error.response.headers.get("retry-after")
        if header_val:
            return float(header_val)
    except Exception:
        pass
 
    match = re.search(r"try again in ([\d.]+)s", str(error))
    if match:
        return float(match.group(1))
 
    return 20.0  # safe fallback if we can't parse a specific wait time
 
 
def call_llm(
    prompt: str,
    max_tokens: int = 2048,
    model: str | None = None,
    reasoning_effort: str = "low",
    max_retries: int = 3,
) -> str:
    """
    Send a single-turn prompt to Groq and return the text response.
 
    Note: openai/gpt-oss-* models on Groq are reasoning models -- their
    internal "thinking" tokens are drawn from the same max_tokens budget
    as the visible answer. On longer prompts a low max_tokens can be
    entirely consumed by reasoning, leaving an EMPTY message.content with
    finish_reason == "length". We default reasoning_effort to "low" to
    keep it focused, and use a generous max_tokens floor so there's room
    left for the actual answer.
 
    Automatically retries on a 429 rate-limit response (up to max_retries
    times), waiting as long as Groq's own error message says to -- a full
    pipeline run easily makes 6+ calls back to back, which can exceed a
    lower-tier account's rolling per-minute token budget partway through
    a run even when each individual call fits comfortably on its own.
 
    Raises on any other client/API error -- callers decide whether to
    fall back or propagate (per each agent's contract in the spec).
    """
    client = _get_client()
    response = None
 
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model or GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            break
        except RateLimitError as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Groq rate limit hit {max_retries + 1} times in a row; "
                    f"giving up. Original error: {e}"
                )
            time.sleep(_seconds_to_wait(e))
 
    choice = response.choices[0]
    content = choice.message.content
    finish_reason = getattr(choice, "finish_reason", "unknown")
 
    if not content:
        raise RuntimeError(
            f"Groq returned empty content (finish_reason={finish_reason}). "
            f"This usually means reasoning tokens consumed the entire "
            f"max_tokens={max_tokens} budget. Try increasing max_tokens "
            f"or lowering reasoning_effort further."
        )
 
    if finish_reason == "length":
        # Content is non-empty but was cut off mid-generation -- e.g. a CSV
        # row truncated partway through an expression. Silently accepting
        # this produces corrupted downstream data that's confusing to
        # debug, so fail loudly here instead with a message that points
        # straight at the fix.
        raise RuntimeError(
            f"Groq response was truncated (finish_reason=length) after "
            f"{max_tokens} max_tokens -- the answer was cut off partway "
            f"through, not just empty. Raise max_tokens for this call."
        )
 
    return content
