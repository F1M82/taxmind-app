export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return res.status(200).end();

  const BASE = process.env.KV_REST_API_URL;
  const TOKEN = process.env.KV_REST_API_TOKEN;

  async function get(key) {
    try {
      const r = await fetch(`${BASE}/get/${key}`, { headers: { Authorization: `Bearer ${TOKEN}` } });
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

  const safeMeta = v => Array.isArray(v) ? v : [];

  async function getAllChunks(id, pages) {
    const results = await Promise.all(
      Array.from({length: pages}, (_, i) => get(`tm_c_${id}_${i}`))
    );
    return results.flatMap(p => Array.isArray(p) ? p : []);
  }

  try {
    // GET without ?chunks=1 → metadata only (fast, for sidebar/UI)
    // GET with ?chunks=1 → full chunks (for search/query)
    if (req.method === 'GET') {
      const withChunks = req.query.chunks === '1';
      const meta = safeMeta(await get('tm_meta'));

      if (!withChunks) {
        // Return metadata only — no chunks, very fast
        return res.status(200).json({ ok: true, documents: meta, count: meta.length });
      }

      // Return full chunks for search
      const docs = await Promise.all(meta.map(async m => {
        const chunks = await getAllChunks(m.id, m.pages || 1);
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
      if (doc) await Promise.all(Array.from({length: doc.pages||1}, (_,i) => del(`tm_c_${id}_${i}`)));
      await set('tm_meta', meta.filter(m => m.id !== id));
      return res.status(200).json({ ok: true });
    }

    return res.status(405).json({ error: 'method not allowed' });
  } catch(e) {
    return res.status(500).json({ error: e.message });
  }
}
