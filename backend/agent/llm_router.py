"""
agent/llm_router.py
─────────────────────────────────────────────────────────────────────────────
TaxMind LLM Router — Switch between OpenAI, Anthropic, and Google Gemini

Usage:
    from agent.llm_router import get_llm

    llm = get_llm("openai")   # GPT-4o-mini
    llm = get_llm("claude")   # Claude Sonnet
    llm = get_llm("gemini")   # Gemini 2.5 Flash

    # Or let the router auto-pick based on task type
    llm = get_llm("auto", task="long_document")   # → Gemini (1M context)
    llm = get_llm("auto", task="reasoning")       # → Claude
    llm = get_llm("auto", task="tool_use")        # → OpenAI
    llm = get_llm("auto", task="cheap")           # → Gemini Flash-Lite
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Model configs ────────────────────────────────────────────────────────────

MODELS = {
    # ── OpenAI ──────────────────────────────────────────────────────────────
    "openai": {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "context_window": 128_000,
        "strengths": ["tool_use", "structured_output", "general"],
        "cost_tier": "medium",
    },
    "openai-powerful": {
        "model": "gpt-4o",
        "provider": "openai",
        "context_window": 128_000,
        "strengths": ["tool_use", "reasoning", "structured_output"],
        "cost_tier": "high",
    },

    # ── Anthropic Claude ─────────────────────────────────────────────────────
    "claude": {
        "model": "claude-sonnet-4-5",
        "provider": "anthropic",
        "context_window": 200_000,
        "strengths": ["reasoning", "long_document", "legal", "tax_analysis"],
        "cost_tier": "medium",
    },
    "claude-lite": {
        "model": "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "context_window": 200_000,
        "strengths": ["cheap", "fast", "general"],
        "cost_tier": "low",
    },

    # ── Google Gemini ────────────────────────────────────────────────────────
    "gemini": {
        "model": "gemini-2.5-flash",
        "provider": "google",
        "context_window": 1_000_000,
        "strengths": ["long_document", "multimodal", "fast", "cheap"],
        "cost_tier": "medium",
    },
    "gemini-lite": {
        "model": "gemini-2.5-flash-lite",
        "provider": "google",
        "context_window": 1_000_000,
        "strengths": ["cheap", "fast", "long_document", "multimodal"],
        "cost_tier": "low",
    },
}

# ─── Auto-routing rules ───────────────────────────────────────────────────────

TASK_ROUTING = {
    "long_document":    "gemini",        # 1M context window
    "multimodal":       "gemini",        # images, audio, video
    "cheap":            "gemini-lite",   # lowest cost
    "fast":             "gemini-lite",   # lowest latency
    "tool_use":         "openai",        # most mature tool ecosystem
    "structured_output":"openai",        # reliable JSON output
    "reasoning":        "claude",        # strong analytical reasoning
    "legal":            "claude",        # careful, nuanced responses
    "tax_analysis":     "claude",        # financial reasoning
    "general":          "openai",        # solid default
}


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_llm(
    provider: str = "openai",
    task: str | None = None,
    temperature: float = 0,
    **kwargs,
):
    """
    Get a LangChain-compatible LLM instance.

    Args:
        provider: "openai" | "claude" | "gemini" | "gemini-lite" |
                  "claude-lite" | "openai-powerful" | "auto"
        task:     Used when provider="auto" to select the best model.
                  Options: long_document, multimodal, cheap, fast,
                           tool_use, structured_output, reasoning,
                           legal, tax_analysis, general
        temperature: 0 = deterministic, higher = more creative
        **kwargs: Passed to the underlying LLM constructor

    Returns:
        LangChain BaseChatModel instance
    """

    # ── Auto routing ──────────────────────────────────────────────────────
    if provider == "auto":
        provider = TASK_ROUTING.get(task or "general", "openai")
        print(f"🔀 Auto-routing task='{task}' → {provider} ({MODELS[provider]['model']})")

    config = MODELS.get(provider)
    if not config:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {list(MODELS.keys())} or 'auto'"
        )

    vendor = config["provider"]

    # ── OpenAI ────────────────────────────────────────────────────────────
    if vendor == "openai":
        _check_key("OPENAI_API_KEY", "OpenAI")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config["model"],
            temperature=temperature,
            **kwargs,
        )

    # ── Anthropic ─────────────────────────────────────────────────────────
    elif vendor == "anthropic":
        _check_key("ANTHROPIC_API_KEY", "Anthropic")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config["model"],
            temperature=temperature,
            **kwargs,
        )

    # ── Google Gemini ─────────────────────────────────────────────────────
    elif vendor == "google":
        _check_key("GOOGLE_API_KEY", "Google Gemini")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config["model"],
            temperature=temperature,
            **kwargs,
        )

    raise ValueError(f"Unsupported vendor: {vendor}")


def _check_key(env_var: str, name: str):
    if not os.getenv(env_var):
        raise EnvironmentError(
            f"{env_var} not set. Add it to your .env file to use {name}."
        )


# ─── Model info helpers ───────────────────────────────────────────────────────

def list_models() -> dict:
    """Print all available models and their properties."""
    return MODELS


def best_for(task: str) -> str:
    """Return the recommended model key for a given task."""
    return TASK_ROUTING.get(task, "openai")


def model_info(provider: str) -> dict:
    """Return config info for a provider."""
    return MODELS.get(provider, {})
