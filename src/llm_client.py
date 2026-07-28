"""LLM Client — Wrapper for Ollama (local open-source models).

Sends prompts to Hermes3 (agentic tasks) or Qwen3 (text generation)
running locally through Ollama. No API key needed.
"""

import json
import time
import requests
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
AGENT_MODEL = os.getenv("AGENT_MODEL", "hermes3")
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen3")


# ---------------------------------------------------------------------------
# Core LLM Functions
# ---------------------------------------------------------------------------

def chat(
    prompt: str,
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    json_mode: bool = False,
) -> str:
    """Send a prompt to the LLM and return the response text.

    Args:
        prompt: The user message to send.
        model: Which Ollama model to use. Defaults to TEXT_MODEL (qwen3).
        system_prompt: Optional system-level instructions.
        temperature: Creativity level (0.0 = deterministic, 1.0 = creative).
        max_tokens: Maximum response length.
        json_mode: If True, request JSON output format.

    Returns:
        The model's response as a string.

    Raises:
        ConnectionError: If Ollama is not running.
    """
    if model is None:
        model = TEXT_MODEL

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    if json_mode:
        payload["format"] = "json"

    start_time = time.time()

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
    except requests.ConnectionError:
        raise ConnectionError(
            "Ollama is not running. Start it with: ollama serve"
        )
    except requests.Timeout:
        raise TimeoutError(
            f"LLM request timed out after 120s (model: {model})"
        )

    elapsed = time.time() - start_time
    result = response.json()
    content = result.get("message", {}).get("content", "").strip()

    # Strip <think>...</think> blocks that Qwen3 sometimes adds
    if "<think>" in content:
        import re
        # Try stripping think blocks
        stripped = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if stripped:
            content = stripped
        else:
            # If stripping leaves nothing, extract text AFTER </think>
            match = re.search(r"</think>\s*(.*)", content, flags=re.DOTALL)
            if match and match.group(1).strip():
                content = match.group(1).strip()
            # else keep original content with tags stripped
            else:
                content = re.sub(r"</?think>", "", content).strip()

    # Log the call (used by the agent logger later)
    _log_call(model, prompt[:100], content[:100], elapsed)

    return content


def chat_json(
    prompt: str,
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.3,
) -> Dict:
    """Send a prompt and parse the response as JSON.

    Uses lower temperature by default for more structured output.
    Falls back to extracting JSON from text if parsing fails.
    """
    if model is None:
        model = AGENT_MODEL  # Hermes3 is better at structured output

    raw = chat(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        json_mode=True,
    )

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Return a wrapper if we can't parse
        return {"raw_response": raw, "parse_error": True}


def is_available() -> bool:
    """Check if Ollama is running and responsive."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def list_models() -> list:
    """Return a list of model names available in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Internal Logging
# ---------------------------------------------------------------------------

_call_log: list = []


def _log_call(model: str, prompt_preview: str, response_preview: str, elapsed: float):
    """Record an LLM call for later inspection."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "prompt": prompt_preview,
        "response": response_preview,
        "elapsed_seconds": round(elapsed, 2),
    }
    _call_log.append(entry)


def get_call_log() -> list:
    """Return the log of all LLM calls made this session."""
    return _call_log.copy()
