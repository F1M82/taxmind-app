const META_KEY = 'taxmind_meta_v1';

async function kvGet(key) {
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  if (!url || !token) throw new Error('KV not configured');
  const res = await fetch(`${url}/get/${encodeURIComponent(key)}`, {
    headers: { Authorization: `Bearer ${token}` }
  });
  if (!res.ok) throw new Error('KV get error ' + res.status);
  const data = await res.json();
  if (!data.result) return null;
  let parsed = data.result;
  if (typeof parsed === 'string') try { parsed = JSON.parse(parsed); } catch(e) {}
  if (typeof parsed === 'string') try { parsed = JSON.parse(parsed); } catch(e) {}
  return parsed;
}

async function kvSet(key, value) {
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  if (!url || !token) throw new Error('KV not configured');
  const res = await fetch(`${url}/set/${encodeURIComponent(key)}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(JSON.stringify(value))
  });
  if (!res.ok) throw new Error('KV set error ' + res.status);
}

async function kvDel(key) {
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  if (!url || !token) return;
  await fetch(`${url}/del/${encodeURIComponent(key)}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` }
  });
}

async function getMeta() {
  const meta = await kvGet(META_KEY);
  return Array.isArray(meta) ? meta : [];
}

async function setMeta(meta) {
  await kvSet(META_KEY, Array.isArray(meta) ? meta : []);
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return res.status(200).end();

  if (req.method === 'GET') {
    try {
      const meta = await getMeta();
      // Load chunks for each doc
      const docs = await Promise.all(meta.map(async m => {
        const chunks = await kvGet('chunks_' + m.id) || [];
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
      const existingIds = new Set(meta.map(d => d.id));
      let added = 0;

      for (const doc of documents) {
        if (!doc.id || !doc.title || doc.title === 'undefined' || existingIds.has(doc.id)) continue;
        
        const chunks = doc.chunks || [];
        const BATCH = 200; // store 200 chunks per key
        
        if (chunks.length <= BATCH) {
          await kvSet('chunks_' + doc.id, chunks);
        } else {
          // Store in multiple keys
          let allChunks = [];
          for (let i = 0; i < chunks.length; i += BATCH) {
            const batch = chunks.slice(i, i + BATCH);
            await kvSet(`chunks_${doc.id}_${Math.floor(i/BATCH)}`, batch);
            allChunks = allChunks.concat(batch);
          }
          await kvSet('chunks_' + doc.id, allChunks.slice(0, BATCH)); // also store first batch under main key
        }

        // Store metadata without chunks
        const { chunks: _, ...docMeta } = doc;
        docMeta.chunkCount = chunks.length;
        meta.push(docMeta);
        existingIds.add(doc.id);
        added++;
      }

      await setMeta(meta);
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
      const updated = meta.filter(d => d.id !== id);
      await setMeta(updated);
      await kvDel('chunks_' + id);
      return res.status(200).json({ ok: true, total: updated.length });
    } catch(e) {
      return res.status(500).json({ error: e.message });
    }
  }

  return res.status(405).json({ error: 'Method not allowed' });
}
