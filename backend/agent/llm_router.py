"""agent/llm_router.py – Multi-provider LLM router with automatic fallback on credit/quota errors."""
import os
from dotenv import load_dotenv
load_dotenv()

MODELS = {
    "claude":           {"model": "claude-haiku-4-5-20251001",   "provider": "anthropic", "cost": 1},
    "claude-powerful":  {"model": "claude-sonnet-4-5",           "provider": "anthropic", "cost": 3},
    "openai":           {"model": "gpt-4o-mini",                 "provider": "openai",    "cost": 2},
    "openai-powerful":  {"model": "gpt-4o",                      "provider": "openai",    "cost": 4},
    "gemini":           {"model": "gemini-2.0-flash",            "provider": "google",    "cost": 1},
    "gemini-lite":      {"model": "gemini-2.0-flash-lite",       "provider": "google",    "cost": 0},
    "groq":             {"model": "llama-3.3-70b-versatile",     "provider": "groq",      "cost": 0},
}

TASK_ROUTING = {
    "general":           "claude",
    "fast":              "claude",
    "cheap":             "claude",
    "tax_analysis":      "claude",
    "legal":             "claude-powerful",
    "reasoning":         "claude-powerful",
    "long_document":     "gemini",
    "multimodal":        "gemini",
    "structured_output": "openai",
    "tool_use":          "openai",
}

# Fallback chain: try in this order when a provider fails
FALLBACK_CHAIN = ["claude", "openai", "groq", "gemini-lite"]


def _has_key(provider):
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "google":    "GOOGLE_API_KEY",
        "groq":      "GROQ_API_KEY",
    }
    val = os.getenv(key_map.get(provider, ""), "").strip()
    return bool(val and len(val) > 10)


def _build_llm(provider, temperature=0, **kwargs):
    """Build a raw LLM instance for the given provider key."""
    config = MODELS.get(provider)
    if not config:
        raise ValueError(f"Unknown provider '{provider}'. Options: {list(MODELS.keys())}")
    vendor = config["provider"]
    if not _has_key(vendor):
        raise EnvironmentError(f"No API key available for provider '{provider}'.")

    if vendor == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config["model"], temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY"), **kwargs
        )
    elif vendor == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config["model"], temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"), **kwargs
        )
    elif vendor == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config["model"], temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"), **kwargs
        )
    elif vendor == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config["model"], temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"), **kwargs
        )
    else:
        raise ValueError(f"Unsupported vendor '{vendor}'.")


# Keywords that indicate a provider should be skipped and next tried
_RETRYABLE_ERRORS = ["credit", "quota", "rate_limit", "rate limit", "429", "billing",
                     "insufficient", "No API key", "No key", "EnvironmentError"]


class FallbackLLM:
    """
    Wraps an LLM and automatically falls back to the next provider in the
    chain when a credit / quota / rate-limit error is encountered.
    Exposes .invoke() so it is a drop-in replacement for any LangChain chat model.
    """

    def __init__(self, primary_provider, temperature=0, **kwargs):
        # Build the ordered chain: primary first, then remaining fallbacks
        self._chain = [primary_provider] + [
            p for p in FALLBACK_CHAIN if p != primary_provider
        ]
        self._temperature = temperature
        self._kwargs = kwargs

    def _is_retryable(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(k.lower() in msg for k in _RETRYABLE_ERRORS)

    def invoke(self, messages, **kw):
        last_exc = None
        for provider in self._chain:
            try:
                llm = _build_llm(provider, temperature=self._temperature, **self._kwargs)
                result = llm.invoke(messages, **kw)
                if provider != self._chain[0]:
                    print(f"✅ FallbackLLM: answered by '{provider}'")
                return result
            except Exception as e:
                if self._is_retryable(e):
                    print(f"⚠️  Provider '{provider}' skipped: {e.__class__.__name__} – {str(e)[:120]}")
                    last_exc = e
                    continue
                raise  # non-retryable error → propagate immediately
        raise EnvironmentError(
            f"All providers exhausted. Last error: {last_exc}"
        )

    # Make it compatible with LangChain's __call__ interface too
    def __call__(self, messages, **kw):
        return self.invoke(messages, **kw)


def get_llm(provider="claude", task=None, temperature=0, **kwargs):
    """
    Returns a FallbackLLM instance.
    - provider: key from MODELS, or "auto" to route by task
    - task: used when provider="auto"
    """
    if provider == "auto":
        provider = TASK_ROUTING.get(task or "general", "claude")
        print(f"🔀 Auto-routing task='{task}' → {provider}")

    if provider not in MODELS:
        raise ValueError(f"Unknown provider '{provider}'. Options: {list(MODELS.keys())}")

    return FallbackLLM(provider, temperature=temperature, **kwargs)


def list_models():
    return MODELS

def list_available():
    return {k: v for k, v in MODELS.items() if _has_key(v["provider"])}

def best_for(task):
    return TASK_ROUTING.get(task, "claude")