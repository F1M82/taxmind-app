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
    await fetch(`${BASE}/set/${key}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(JSON.stringify(val))
    });
  }

  async function del(key) {
    await fetch(`${BASE}/del/${key}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' }
    });
  }

  function safeMeta(v) {
    return Array.isArray(v) ? v : [];
  }

  try {
    if (req.method === 'GET') {
      const meta = safeMeta(await get('tm_meta'));
      // Fetch ALL chunk pages in parallel for speed
      const docs = await Promise.all(meta.map(async m => {
        const pages = m.pages || 1;
        // Fetch all pages in parallel
        const pagePromises = [];
        for (let i = 0; i < pages; i++) {
          pagePromises.push(get(`tm_c_${m.id}_${i}`));
        }
        const pageResults = await Promise.all(pagePromises);
        const chunks = pageResults.flatMap(p => Array.isArray(p) ? p : []);
        return { ...m, chunks, chunkCount: chunks.length };
      }));
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
      if (doc) {
        const delPromises = [];
        for (let i = 0; i < (doc.pages || 1); i++) delPromises.push(del(`tm_c_${id}_${i}`));
        await Promise.all(delPromises);
      }
      await set('tm_meta', meta.filter(m => m.id !== id));
      return res.status(200).json({ ok: true });
    }

    return res.status(405).json({ error: 'method not allowed' });
  } catch(e) {
    return res.status(500).json({ error: e.message });
  }
}
