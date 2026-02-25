// Web Search API for TaxMind
// Uses Google Custom Search JSON API — 100 free searches/day (no credit card needed)
//
// SETUP (10 minutes, completely free):
// 1. Go to https://programmablesearchengine.google.com → Create search engine
//    - Name: TaxMind, Search the entire web: ON → Create → copy Search Engine ID (cx)
// 2. Go to https://console.cloud.google.com → APIs & Services → Library
//    → search "Custom Search API" → Enable it
//    → Credentials → Create Credentials → API Key → copy it
// 3. Add to Vercel environment variables:
//    GOOGLE_SEARCH_API_KEY = your API key
//    GOOGLE_SEARCH_CX     = your Search Engine ID

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const apiKey = process.env.GOOGLE_SEARCH_API_KEY;
  const cx     = process.env.GOOGLE_SEARCH_CX;

  if (!apiKey || !cx) {
    return res.status(200).json({ ok: false, results: [], message: 'Web search not configured' });
  }

  const { query } = req.body;
  if (!query) return res.status(400).json({ error: 'query required' });

  try {
    const scopedQuery = `${query} (site:incometaxindia.gov.in OR site:cbdt.gov.in OR site:itatonline.org OR site:indiankanoon.org OR site:taxmann.com OR "income tax" India)`;
    const url = `https://www.googleapis.com/customsearch/v1?key=${apiKey}&cx=${cx}&q=${encodeURIComponent(scopedQuery)}&num=6&dateRestrict=y2`;

    const response = await fetch(url);
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error?.message || 'Google Search error ' + response.status);
    }

    const data = await response.json();
    const results = (data.items || []).map(item => ({
      title: item.title,
      url: item.link,
      snippet: item.snippet || '',
      published: item.pagemap?.metatags?.[0]?.['article:published_time'] || '',
    }));

    return res.status(200).json({ ok: true, results });
  } catch (err) {
    return res.status(200).json({ ok: false, results: [], message: err.message });
  }
}
