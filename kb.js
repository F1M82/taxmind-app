const KB_KEY = 'taxmind_kb_v1';

async function redisGet(key) {
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;
  if (!url || !token) return null;
  const res = await fetch(`${url}/get/${key}`, {
    headers: { Authorization: `Bearer ${token}` }
  });
  if (!res.ok) return null;
  const data = await res.json();
  if (!data.result) return null;
  try { return JSON.parse(data.result); } catch(e) { return null; }
}

async function redisSet(key, value) {
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;
  if (!url || !token) throw new Error('Upstash not configured. Go to Vercel Storage and connect Upstash Redis to this project.');
  const res = await fetch(`${url}/set/${key}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(JSON.stringify(value))
  });
  if (!res.ok) throw new Error('Redis write failed: ' + res.status);
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return res.status(200).end();

  if (req.method === 'GET') {
    try {
      const kb = await redisGet(KB_KEY) || [];
      return res.status(200).json({ ok: true, documents: kb, count: kb.length });
    } catch (e) {
      return res.status(500).json({ error: e.message });
    }
  }

  if (req.method === 'POST') {
    try {
      const { documents } = req.body;
      if (!documents || !Array.isArray(documents)) {
        return res.status(400).json({ error: 'documents array required' });
      }
      const existing = await redisGet(KB_KEY) || [];
      const existingIds = new Set(existing.map(d => d.id));
      const newDocs = documents.filter(d => !existingIds.has(d.id));
      const merged = [...existing, ...newDocs];
      const serialized = JSON.stringify(merged);
      let toStore = merged;
      if (serialized.length > 900000) {
        toStore = merged.map(d => ({ ...d, chunks: (d.chunks || []).slice(0, 30) }));
      }
      await redisSet(KB_KEY, toStore);
      return res.status(200).json({ ok: true, added: newDocs.length, total: merged.length });
    } catch (e) {
      return res.status(500).json({ error: e.message });
    }
  }

  if (req.method === 'DELETE') {
    try {
      const { id } = req.query;
      if (!id) return res.status(400).json({ error: 'id required' });
      const existing = await redisGet(KB_KEY) || [];
      const updated = existing.filter(d => d.id !== id);
      await redisSet(KB_KEY, updated);
      return res.status(200).json({ ok: true, total: updated.length });
    } catch (e) {
      return res.status(500).json({ error: e.message });
    }
  }

  return res.status(405).json({ error: 'Method not allowed' });
}
