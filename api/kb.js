export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return res.status(200).end();

  const URL = process.env.KV_REST_API_URL;
  const TOKEN = process.env.KV_REST_API_TOKEN;

  async function get(key) {
    const r = await fetch(`${URL}/get/${key}`, { headers: { Authorization: `Bearer ${TOKEN}` } });
    const d = await r.json();
    if (!d.result) return null;
    try { return JSON.parse(d.result); } catch(e) { return null; }
  }

  async function set(key, val) {
    await fetch(`${URL}/set/${key}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(JSON.stringify(val))
    });
  }

  async function del(key) {
    await fetch(`${URL}/del/${key}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' }
    });
  }

  try {
    if (req.method === 'GET') {
      const meta = await get('tm_meta') || [];
      const docs = [];
      for (const m of meta) {
        const pages = m.pages || 1;
        let chunks = [];
        for (let i = 0; i < pages; i++) {
          const p = await get(`tm_c_${m.id}_${i}`) || [];
          chunks = chunks.concat(p);
        }
        docs.push({ ...m, chunks, chunkCount: chunks.length });
      }
      return res.status(200).json({ ok: true, documents: docs, count: docs.length });
    }

    if (req.method === 'POST') {
      const { documents } = req.body;
      if (!Array.isArray(documents)) return res.status(400).json({ error: 'need documents array' });
      
      let meta = await get('tm_meta') || [];
      
      for (const doc of documents) {
        if (!doc.id || !doc.title || doc.title === 'undefined') continue;
        const chunks = Array.isArray(doc.chunks) ? doc.chunks : [];
        const page = doc.chunkPage || 0;
        const pages = doc.totalPages || 1;
        
        await set(`tm_c_${doc.id}_${page}`, chunks);
        
        const exists = meta.find(m => m.id === doc.id);
        if (exists) {
          exists.pages = pages;
          exists.chunkCount = doc.chunkCount || chunks.length;
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
      let meta = await get('tm_meta') || [];
      const doc = meta.find(m => m.id === id);
      if (doc) {
        for (let i = 0; i < (doc.pages || 1); i++) await del(`tm_c_${id}_${i}`);
      }
      meta = meta.filter(m => m.id !== id);
      await set('tm_meta', meta);
      return res.status(200).json({ ok: true, total: meta.length });
    }

    return res.status(405).json({ error: 'method not allowed' });
  } catch(e) {
    return res.status(500).json({ error: e.message });
  }
}
