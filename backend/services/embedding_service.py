import os
from typing import List

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

def get_embeddings(texts: List[str]) -> List[List[float]]:
    if EMBEDDING_PROVIDER == "local":
        return _local_embeddings(texts)
    return _openai_embeddings(texts)

def get_single_embedding(text: str) -> List[float]:
    return get_embeddings([text])[0]

def _local_embeddings(texts: List[str]) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(texts).tolist()

def _openai_embeddings(texts: List[str]) -> List[List[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    results = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        response = client.embeddings.create(input=batch, model=model)
        results.extend([item.embedding for item in response.data])
    return results
