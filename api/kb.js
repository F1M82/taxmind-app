const META_KEY = 'tm_meta_v2';
const PAGE_SIZE = 50;

async function kvGet(key) {
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  const res = await fetch(`${url}/get/${encodeURIComponent(key)}`, {
    headers: { Authorization: `Bearer ${token}` }
  });
  const data = await res.json();
  if (!data.result) return null;
  let v = data.result;
  if (typeof v === 'string') try { v = JSON.parse(v); } catch(e){}
  if (typeof v === 'string') try { v = JSON.parse(v); } catch(e){}
  return v;
}

async function kvSet(key, value) {
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  const res = await fetch(`${url}/set/${encodeURIComponent(key)}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(JSON.stringify(value))
  });
  if (!res.ok) throw new Error('KV write failed ' + res.status);
}

async function kvDel(key) {
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  await fetch(`${url}/del/${encodeURIComponent(key)}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }
  });
}

async function getMeta() {
  const v = await kvGet(META_KEY);
  return Array.isArray(v) ? v : [];
}

async function getChunks(docId, totalPages) {
  const chunks = [];
  for (let i = 0; i < totalPages; i++) {
    const page = await kvGet(`tm_c_${docId}_${i}`);
    if (Array.isArray(page)) chunks.push(...page);
  }
  return chunks;
}

async function delChunks(docId, totalPages) {
  for (let i = 0; i < totalPages; i++) await kvDel(`tm_c_${docId}_${i}`);
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return res.status(200).end();

  if (req.method === 'GET') {
    try {
      const meta = await getMeta();
      const docs = await Promise.all(meta.map(async m => {
        const chunks = await getChunks(m.id, m.totalPages || 0);
        return { ...m, chunks, chunkCount: chunks.length };
      }));
      return res.status(200).json({ ok: true, documents: docs, count: docs.length });
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

      const meta = await getMeta();
      let added = 0;

      for (const doc of documents) {
        if (!doc.id || !doc.title || doc.title === 'undefined') continue;

        const chunks = Array.isArray(doc.chunks) ? doc.chunks : [];
        const chunkPage = doc.chunkPage ?? 0;
        const totalPages = doc.totalPages ?? 1;
        const existingIdx = meta.findIndex(d => d.id === doc.id);

        // Store this page of chunks
        await kvSet(`tm_c_${doc.id}_${chunkPage}`, chunks);

        if (existingIdx >= 0) {
          // Update existing doc metadata
          meta[existingIdx].totalPages = totalPages;
          meta[existingIdx].chunkCount = doc.chunkCount || chunks.length;
        } else {
          // New doc — add metadata
          const { chunks: _, chunkPage: __, totalPages: ___, ...docMeta } = doc;
          meta.push({ ...docMeta, totalPages, chunkCount: doc.chunkCount || chunks.length });
          added++;
        }
      }

      await kvSet(META_KEY, meta);
      return res.status(200).json({ ok: true, added, total: meta.length });
    } catch(e) {
      return res.status(500).json({ error: e.message });
    }
  }

  if (req.method === 'DELETE') {
    try {
      const { id } = req.query;
      if (!id) return res.status(400).json({ error: 'id required' });
      const meta = await getMeta();
      const doc = meta.find(d => d.id === id);
      if (doc) await delChunks(id, doc.totalPages || 0);
      const updated = meta.filter(d => d.id !== id);
      await kvSet(META_KEY, updated);
      return res.status(200).json({ ok: true, total: updated.length });
    } catch(e) {
      return res.status(500).json({ error: e.message });
    }
  }

  return res.status(405).json({ error: 'Method not allowed' });
}
