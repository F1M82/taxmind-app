export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return res.status(200).end();

  const BASE = process.env.KV_REST_API_URL;
  const TOKEN = process.env.KV_REST_API_TOKEN;

  async function get(key) {
    try {
      const r = await fetch(`${BASE}/get/${key}`, {
        headers: { Authorization: `Bearer ${TOKEN}` }
      });
      const d = await r.json();
      if (!d || !d.result) return null;
      let v = d.result;
      if (typeof v === 'string') { try { v = JSON.parse(v); } catch(e) { return null; } }
      if (typeof v === 'string') { try { v = JSON.parse(v); } catch(e) { return null; } }
      return v;
    } catch(e) { return null; }
  }

  async function set(key, val) {
    try {
      await fetch(`${BASE}/set/${key}`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(JSON.stringify(val))
      });
    } catch(e) { throw new Error('set failed: ' + e.message); }
  }

  async function del(key) {
    try {
      await fetch(`${BASE}/del/${key}`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' }
      });
    } catch(e) {}
  }

  function safeMeta(v) {
    if (Array.isArray(v)) return v;
    return [];
  }

  try {
    if (req.method === 'GET') {
      const meta = safeMeta(await get('tm_meta'));
      const docs = [];
      for (const m of meta) {
        let chunks = [];
        for (let i = 0; i < (m.pages || 1); i++) {
          const p = await get(`tm_c_${m.id}_${i}`);
          if (Array.isArray(p)) chunks = chunks.concat(p);
        }
        docs.push({ ...m, chunks, chunkCount: chunks.length });
      }
      return res.status(200).json({ ok: true, documents: docs, count: docs.length });
    }

    if (req.method === 'POST') {
      const { documents } = req.body;
      if (!Array.isArray(documents)) return res.status(400).json({ error: 'need documents array' });

      const meta = safeMeta(await get('tm_meta'));

      for (const doc of documents) {
        if (!doc.id || !doc.title || doc.title === 'undefined') continue;
        const chunks = Array.isArray(doc.chunks) ? doc.chunks : [];
        const page = typeof doc.chunkPage === 'number' ? doc.chunkPage : 0;
        const pages = typeof doc.totalPages === 'number' ? doc.totalPages : 1;

        await set(`tm_c_${doc.id}_${page}`, chunks);

        const idx = meta.findIndex(m => m.id === doc.id);
        if (idx >= 0) {
          meta[idx].pages = pages;
          meta[idx].chunkCount = doc.chunkCount || chunks.length;
        } else {
          const { chunks: _c, chunkPage: _p, totalPages: _t, ...clean } = doc;
          meta.push({ ...clean, pages, chunkCount: doc.chunkCount || chunks.length });
        }
      }

      await set('tm_meta', meta);
      return res.status(200).json({ ok: true, total: meta.length });
    }

    if (req.method === 'DELETE') {
      const { id } = req.query;
      if (!id) return res.status(400).json({ error: 'need id' });
      const meta = safeMeta(await get('tm_meta'));
      const doc = meta.find(m => m.id === id);
      if (doc) for (let i = 0; i < (doc.pages || 1); i++) await del(`tm_c_${id}_${i}`);
      await set('tm_meta', meta.filter(m => m.id !== id));
      return res.status(200).json({ ok: true });
    }

    return res.status(405).json({ error: 'method not allowed' });
  } catch(e) {
    return res.status(500).json({ error: e.message });
  }
}
