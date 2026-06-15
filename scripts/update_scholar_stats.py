#!/usr/bin/env python3
"""Local runner for the GS scraping logic. Same as the GH Actions step.

If SERPAPI_KEY env var is set, uses SerpApi (reliable, paid quota).
Otherwise falls back to scholarly (free, may be blocked from CI IPs).
"""
import json
import os
import re
import sys
import time
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path

AUTHOR_ID = 'IjoWeIMAAAAJ'
SERPAPI_KEY = os.environ.get('SERPAPI_KEY')
ROOT = Path(__file__).resolve().parent.parent
PAPERS_TOML = ROOT / 'data' / 'papers.toml'
OUT_DIR = ROOT / 'data' / 'gs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCHOLARLY_PAPERS_KEY = 'public' + 'ations'

def log(msg):
    print(msg, flush=True)

def norm(s):
    return re.sub(r'\s+', ' ', s.strip().lower())

with PAPERS_TOML.open('rb') as f:
    papers = tomllib.load(f).get('papers', [])
wanted = {norm(p['title']): p['gs_id'] for p in papers if p.get('gs_id')}
log(f"Tracking {len(wanted)} papers: {sorted(wanted.values())}")

def fetch_serpapi():
    log("  fetching via SerpApi...")
    params = {
        'engine': 'google_scholar_author',
        'author_id': AUTHOR_ID,
        'hl': 'en',
        'num': '100',
        'api_key': SERPAPI_KEY,
    }
    url = 'https://serpapi.com/search.json?' + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:
        payload = json.loads(r.read())
    if payload.get('error'):
        raise RuntimeError(f"SerpApi error: {payload['error']}")

    total = 0
    for row in payload.get('cited_by', {}).get('table', []):
        cits = row.get('citations', {})
        if cits.get('all') is not None:
            total = int(cits['all'])
            break

    by_title = {}
    for art in payload.get('articles', []):
        title = norm(str(art.get('title', '')))
        cited = art.get('cited_by', {})
        n = int((cited.get('value') if isinstance(cited, dict) else cited) or 0)
        if title:
            by_title[title] = n
    return total, by_title

def fetch_scholarly():
    log("  fetching via scholarly...")
    try:
        from scholarly import scholarly
    except ImportError:
        raise RuntimeError("scholarly not installed; pip install scholarly")
    author = scholarly.search_author_id(AUTHOR_ID)
    author = scholarly.fill(author, sections=['basics', 'indices', SCHOLARLY_PAPERS_KEY])
    total = author.get('citedby', 0)
    by_title = {}
    for pub in author.get(SCHOLARLY_PAPERS_KEY, []):
        title = pub.get('bib', {}).get('title', '')
        if title:
            by_title[norm(title)] = pub.get('num_citations', 0) or 0
    return total, by_title

def write_badge(name, citations):
    payload = {
        "schemaVersion": 1, "label": "citations", "message": str(citations),
        "color": "1a5276", "namedLogo": "Google Scholar", "logoColor": "white",
    }
    (OUT_DIR / f"{name}.json").write_text(json.dumps(payload))

last_error = None
for attempt in range(1, 3):
    try:
        log(f"Attempt {attempt}...")
        if SERPAPI_KEY:
            total, by_title = fetch_serpapi()
        else:
            total, by_title = fetch_scholarly()

        if not total:
            raise RuntimeError("fetch returned empty citedby")

        matched = 0
        for title_key, gs_id in wanted.items():
            if title_key in by_title:
                write_badge(gs_id, by_title[title_key])
                matched += 1
                log(f"  matched {gs_id}: {by_title[title_key]} citations")
            else:
                log(f"  WARN no match for {gs_id} (title: {title_key!r})")

        write_badge('total', total)
        log(f"Total: {total}, matched {matched}/{len(wanted)} papers")
        break
    except Exception as exc:
        last_error = exc
        log(f"Attempt {attempt} failed: {exc}")
        if attempt < 2:
            time.sleep(10)
else:
    log(f"All attempts failed. Last error: {last_error}")
    sys.exit(1)
