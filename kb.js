// Shared Knowledge Base API
// Uses Vercel KV (free Redis-compatible store) to share KB across all team members
// KV_REST_API_URL and KV_REST_API_TOKEN are auto-set when you add Vercel KV storage

const KB_KEY = 'taxmind_kb_v1';
const MAX_KB_SIZE = 4 * 1024 * 1024; // 4MB KV limit per value

async function kvGet(key) {
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  if (!url || !token) return null;
  const res = await fetch(`${url}/get/${key}`, {
    headers: { Authorization: `Bearer ${token}` }
  });
  if (!res.ok) return null;
  const data = await res.json();
  return data.result ? JSON.parse(data.result) : null;
}

async function kvSet(key, value) {
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  if (!url || !token) throw new Error('Vercel KV not configured');
  const res = await fetch(`${url}/set/${key}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ value: JSON.stringify(value) })
  });
  if (!res.ok) throw new Error('KV write failed');
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return res.status(200).end();

  // GET — return current knowledge base index (without chunks, just metadata)
  if (req.method === 'GET') {
    try {
      const kb = await kvGet(KB_KEY) || [];
      // Return metadata only (chunks stay in localStorage on upload device)
      // But for shared retrieval, we store full chunks too - just strip for listing
      const listing = kb.map(({ chunks, ...meta }) => meta);
      return res.status(200).json({ ok: true, documents: listing, count: kb.length });
    } catch (e) {
      return res.status(500).json({ error: e.message });
    }
  }

  // POST — add documents to shared KB
  if (req.method === 'POST') {
    try {
      const { documents } = req.body;
      if (!documents || !Array.isArray(documents)) {
        return res.status(400).json({ error: 'documents array required' });
      }

      const existing = await kvGet(KB_KEY) || [];
      const existingIds = new Set(existing.map(d => d.id));
      const newDocs = documents.filter(d => !existingIds.has(d.id));
      const merged = [...existing, ...newDocs];

      // Check size limit
      const serialized = JSON.stringify(merged);
      if (serialized.length > MAX_KB_SIZE) {
        // Store without chunk content if too large (fallback to metadata-only)
        const slim = merged.map(d => ({ ...d, chunks: d.chunks?.slice(0, 20) }));
        await kvSet(KB_KEY, slim);
        return res.status(200).json({ ok: true, added: newDocs.length, warning: 'Large KB — some chunks trimmed' });
      }

      await kvSet(KB_KEY, merged);
      return res.status(200).json({ ok: true, added: newDocs.length, total: merged.length });
    } catch (e) {
      return res.status(500).json({ error: e.message });
    }
  }

  // DELETE — remove a document
  if (req.method === 'DELETE') {
    try {
      const { id } = req.query;
      const existing = await kvGet(KB_KEY) || [];
      const updated = existing.filter(d => d.id !== id);
      await kvSet(KB_KEY, updated);
      return res.status(200).json({ ok: true, total: updated.length });
    } catch (e) {
      return res.status(500).json({ error: e.message });
    }
  }

  return res.status(405).json({ error: 'Method not allowed' });
}
