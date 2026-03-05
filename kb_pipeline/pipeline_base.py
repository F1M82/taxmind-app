import json, hashlib, os, argparse
from pathlib import Path
from datetime import datetime

for d in ["review_queue","approved","rejected","logs"]:
    Path(d).mkdir(exist_ok=True)

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL","http://localhost:9200")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX","taxmind-kb")

def doc_id(t): return hashlib.md5(t.encode()).hexdigest()[:12]
def save_chunks(chunks, source):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("review_queue") / f"{source}_{ts}.json"
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved {len(chunks)} chunks to {out}")
def make_chunk(text, source, meta):
    return {"id":doc_id(text),"text":text.strip(),"source":source,"metadata":{**meta,"extracted_at":datetime.now().isoformat()},"status":"pending"}
