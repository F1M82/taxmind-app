"""services/monitoring.py — Langfuse monitoring and tracing"""
import os, time, functools
from langfuse import Langfuse

PK   = os.getenv("LANGFUSE_PUBLIC_KEY", "")
SK   = os.getenv("LANGFUSE_SECRET_KEY", "")
HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

_lf = None
def _get():
    global _lf
    if _lf is None and PK and SK:
        try: _lf = Langfuse(public_key=PK, secret_key=SK, host=HOST)
        except Exception as e: print(f"⚠️ Langfuse init failed: {e}")
    return _lf

def log_agent_run(query, result, provider, user_id=None, session_id=None):
    lf = _get()
    if not lf: return
    try:
        lf.trace(
            name="taxmind_agent_run", user_id=user_id, session_id=session_id,
            input={"query": query}, output={"answer": result.get("answer", "")[:500]},
            metadata={"provider": provider, "cached": result.get("cached", False),
                      "tools_used": result.get("tools_used", []),
                      "retry_count": result.get("retry_count", 0)}
        )
        lf.flush()
    except Exception as e: print(f"⚠️ Langfuse log failed: {e}")

def track(name):
    """Decorator to trace any function call."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            lf = _get()
            start = time.time()
            result = func(*args, **kwargs)
            if lf:
                try:
                    t = lf.trace(name=name)
                    t.span(name=f"{name}_call", input=str(args)[:200],
                           output=str(result)[:500],
                           metadata={"latency": round(time.time()-start, 3)})
                    lf.flush()
                except: pass
            return result
        return wrapper
    return decorator

def dashboard_url(): return f"{HOST}/traces"
