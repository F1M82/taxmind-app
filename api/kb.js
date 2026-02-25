const KB_KEY = 'taxmind_kb_v1';

async function redisGet(key) {
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
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
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  if (!url || !token) throw new Error('KV not configured');
  const serialized = JSON.stringify(value);
  // Split into multiple keys if too large
  if (serialized.length > 400000) {
    // Store in parts
    const parts = [];
    const chunkSize = 350000;
    for (let i = 0; i < serialized.length; i += chunkSize) {
      parts.push(serialized.slice(i, i + chunkSize));
    }
    await redisSetRaw(key + '_parts', parts.length.toString(), token, url);
    for (let i = 0; i < parts.length; i++) {
      await redisSetRaw(key + '_part_' + i, parts[i], token, url);
    }
    await redisSetRaw(key + '_split', 'true', token, url);
    return;
  }
  await redisSetRaw(key, serialized, token, url);
  await redisSetRaw(key + '_split', 'false', token, url);
}

async function redisSetRaw(key, value, token, url) {
  const res = await fetch(`${url}/set/${key}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(value)
  });
  if (!res.ok) throw new Error('Redis write failed: ' + res.status);
}

async function redisGetFull(key) {
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  if (!url || !token) return null;

  // Check if split
  const splitRes = await fetch(`${url}/get/${key}_split`, { headers: { Authorization: `Bearer ${token}` } });
  const splitData = await splitRes.json();
  const isSplit = splitData.result === 'true';

  if (isSplit) {
    const partsRes = await fetch(`${url}/get/${key}_parts`, { headers: { Authorization: `Bearer ${token}` } });
    const partsData = await partsRes.json();
    const numParts = parseInt(partsData.result || '0');
    let combined = '';
    for (let i = 0; i < numParts; i++) {
      const partRes = await fetch(`${url}/get/${key}_part_${i}`, { headers: { Authorization: `Bearer ${token}` } });
      const partData = await partRes.json();
      combined += (partData.result || '');
    }
    try { return JSON.parse(combined); } catch(e) { return null; }
  }

  return redisGet(key);
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return res.status(200).end();

  if (req.method === 'GET') {
    try {
      const kb = await redisGetFull(KB_KEY) || [];
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

      const existing = await redisGetFull(KB_KEY) || [];

      // Handle batched uploads — merge chunks for same doc id
      const merged = [...existing];
      for (const incoming of documents) {
        const existingIdx = merged.findIndex(d => d.id === incoming.id);
        if (incoming.totalBatches && incoming.totalBatches > 1) {
          // Batched upload — append chunks
          if (existingIdx >= 0) {
            merged[existingIdx].chunks = [...(merged[existingIdx].chunks || []), ...(incoming.chunks || [])];
            merged[existingIdx].chunkCount = merged[existingIdx].chunks.length;
          } else {
            merged.push({ ...incoming, chunkBatch: undefined, totalBatches: undefined });
          }
        } else {
          // Single upload
          if (existingIdx < 0) merged.push(incoming);
        }
      }

      await redisSet(KB_KEY, merged);
      return res.status(200).json({ ok: true, added: documents.length, total: merged.length });
    } catch (e) {
      return res.status(500).json({ error: e.message });
    }
  }

  if (req.method === 'DELETE') {
    try {
      const { id } = req.query;
      if (!id) return res.status(400).json({ error: 'id required' });
      const existing = await redisGetFull(KB_KEY) || [];
      const updated = existing.filter(d => d.id !== id);
      await redisSet(KB_KEY, updated);
      return res.status(200).json({ ok: true, total: updated.length });
    } catch (e) {
      return res.status(500).json({ error: e.message });
    }
  }

  return res.status(405).json({ error: 'Method not allowed' });
}
