"""agent/llm_router.py — Multi-provider LLM router. Primary: Claude → OpenAI → Gemini"""
import os
from dotenv import load_dotenv
load_dotenv()

MODELS = {
    "claude":          {"model": "claude-haiku-4-5-20251001",  "provider": "anthropic", "cost": 1},
    "claude-powerful": {"model": "claude-sonnet-4-5",          "provider": "anthropic", "cost": 3},
    "openai":          {"model": "gpt-4o-mini",                "provider": "openai",    "cost": 2},
    "openai-powerful": {"model": "gpt-4o",                     "provider": "openai",    "cost": 4},
    "gemini":          {"model": "gemini-2.0-flash",           "provider": "google",    "cost": 1},
    "gemini-lite":     {"model": "gemini-2.0-flash-lite",      "provider": "google",    "cost": 0},
}

TASK_ROUTING = {
    "general":          "claude",
    "fast":             "claude",
    "cheap":            "claude",
    "tax_analysis":     "claude",
    "legal":            "claude-powerful",
    "reasoning":        "claude-powerful",
    "long_document":    "gemini",        # Gemini has 1M context — better for large docs
    "multimodal":       "gemini",
    "structured_output":"openai",
    "tool_use":         "openai",
}

# Fallback chain: Claude first, then OpenAI, then Gemini
FALLBACK_CHAIN = ["claude", "openai", "gemini", "gemini-lite"]

def _has_key(provider):
    key_map = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}
    val = os.getenv(key_map.get(provider, ""), "").strip()
    return bool(val and len(val) > 10)

def get_llm(provider="claude", task=None, temperature=0, **kwargs):
    if provider == "auto":
        provider = TASK_ROUTING.get(task or "general", "claude")
        print(f"🔀 Auto-routing task='{task}' → {provider}")

    config = MODELS.get(provider)
    if not config:
        raise ValueError(f"Unknown provider '{provider}'. Options: {list(MODELS.keys())}")

    vendor = config["provider"]
    if not _has_key(vendor):
        print(f"⚠️  No key for '{provider}', trying fallback chain...")
        for fallback in FALLBACK_CHAIN:
            fb_vendor = MODELS[fallback]["provider"]
            if _has_key(fb_vendor):
                print(f"✅ Falling back to '{fallback}'")
                provider, config, vendor = fallback, MODELS[fallback], fb_vendor
                break
        else:
            raise EnvironmentError("No API keys available for any provider.")

    if vendor == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=config["model"], temperature=temperature,
                             api_key=os.getenv("ANTHROPIC_API_KEY"), **kwargs)
    elif vendor == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config["model"], temperature=temperature,
                          api_key=os.getenv("OPENAI_API_KEY"), **kwargs)
    elif vendor == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=config["model"], temperature=temperature,
                                      google_api_key=os.getenv("GOOGLE_API_KEY"), **kwargs)

def list_models():    return MODELS
def list_available(): return {k: v for k, v in MODELS.items() if _has_key(v["provider"])}
def best_for(task):   return TASK_ROUTING.get(task, "claude")