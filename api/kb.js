const KB_KEY = 'taxmind_kb_v1';

async function kv(path, opts={}) {
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  if (!url || !token) throw new Error('KV not configured');
  const res = await fetch(`${url}${path}`, {
    ...opts,
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', ...(opts.headers||{}) }
  });
  if (!res.ok) throw new Error('KV error ' + res.status);
  return res.json();
}

async function getKB() {
  try {
    const data = await kv(`/get/${KB_KEY}`);
    if (!data.result) return [];
    return JSON.parse(data.result);
  } catch(e) { return []; }
}

async function setKB(value) {
  const serialized = JSON.stringify(value);
  await kv(`/set/${KB_KEY}`, {
    method: 'POST',
    body: JSON.stringify(serialized)
  });
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return res.status(200).end();

  if (req.method === 'GET') {
    try {
      const kb = await getKB();
      return res.status(200).json({ ok: true, documents: kb, count: kb.length });
    } catch(e) {
      return res.status(500).json({ error: e.message });
    }
  }

  if (req.method === 'POST') {
    try {
      const { documents } = req.body;
      if (!documents || !Array.isArray(documents)) {
        return res.status(400).json({ error: 'documents array required' });
      }
      const existing = await getKB();
      const existingIds = new Set(existing.map(d => d.id));
      // Only add truly new documents (skip duplicates by id)
      const newDocs = documents.filter(d => d.id && !existingIds.has(d.id) && d.title && d.title !== 'undefined');
      const merged = [...existing, ...newDocs];
      await setKB(merged);
      return res.status(200).json({ ok: true, added: newDocs.length, total: merged.length });
    } catch(e) {
      return res.status(500).json({ error: e.message });
    }
  }

  if (req.method === 'DELETE') {
    try {
      const { id } = req.query;
      if (!id) return res.status(400).json({ error: 'id required' });
      const existing = await getKB();
      const updated = existing.filter(d => d.id !== id);
      await setKB(updated);
      return res.status(200).json({ ok: true, total: updated.length });
    } catch(e) {
      return res.status(500).json({ error: e.message });
    }
  }

  return res.status(405).json({ error: 'Method not allowed' });
}
