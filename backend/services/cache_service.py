"""services/cache_service.py — Redis semantic + exact caching"""
import os, json, hashlib
import numpy as np
import redis
from sentence_transformers import SentenceTransformer

REDIS_URL   = os.getenv("REDIS_URL", "redis://localhost:6379")
TTL         = int(os.getenv("CACHE_TTL_SECONDS", 3600))
SIM_THRESH  = float(os.getenv("CACHE_SIM_THRESHOLD", 0.92))
MODEL_NAME  = "all-MiniLM-L6-v2"

_r, _m = None, None
def _redis():
    global _r
    if _r is None: _r = redis.from_url(REDIS_URL, decode_responses=False)
    return _r
def _model():
    global _m
    if _m is None: _m = SentenceTransformer(MODEL_NAME)
    return _m

def _key(q):     return f"taxmind:exact:{hashlib.md5(q.lower().strip().encode()).hexdigest()}"
def _emb_key(q): return f"taxmind:emb:{hashlib.md5(q.lower().strip().encode()).hexdigest()}"

def get_exact(query):
    try:
        v = _redis().get(_key(query))
        return json.loads(v) if v else None
    except: return None

def set_exact(query, result):
    try: _redis().setex(_key(query), TTL, json.dumps(result))
    except: pass

def get_semantic(query):
    try:
        r, q_emb = _redis(), _model().encode([query], normalize_embeddings=True)[0]
        best_score, best_key = 0.0, None
        for k in r.scan_iter("taxmind:emb:*"):
            data = r.get(k)
            if not data: continue
            entry = json.loads(data)
            score = float(np.dot(q_emb, np.array(entry["embedding"])))
            if score > best_score: best_score, best_key = score, entry.get("result_key")
        if best_score >= SIM_THRESH and best_key:
            v = r.get(best_key)
            if v:
                print(f"💾 Semantic cache hit (similarity: {best_score:.3f})")
                return json.loads(v)
    except: pass
    return None

def set_semantic(query, result):
    try:
        r, q_emb = _redis(), _model().encode([query], normalize_embeddings=True)[0]
        rk = _key(query)
        r.setex(rk, TTL, json.dumps(result))
        r.setex(_emb_key(query), TTL, json.dumps({"embedding": q_emb.tolist(), "result_key": rk, "query": query}))
    except: pass

def get_model_result(key):
    try:
        v = _redis().get(f"taxmind:model:{key}")
        return json.loads(v) if v else None
    except: return None

def cache_model_result(key, result, ttl=300):
    try: _redis().setex(f"taxmind:model:{key}", ttl, json.dumps(result))
    except: pass

def cache_stats():
    try:
        r, info = _redis(), _redis().info("stats")
        hits, misses = info.get("keyspace_hits", 0), info.get("keyspace_misses", 0)
        return {"total_keys": r.dbsize(), "hits": hits, "misses": misses,
                "hit_rate": round(hits / max(hits + misses, 1), 4)}
    except Exception as e: return {"error": str(e)}

def flush_cache():
    try:
        r, keys = _redis(), list(_redis().scan_iter("taxmind:*"))
        if keys: r.delete(*keys)
        return {"deleted": len(keys)}
    except Exception as e: return {"error": str(e)}
