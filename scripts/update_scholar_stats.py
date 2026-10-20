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
import tempfile
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path

AUTHOR_ID = 'IjoWeIMAAAAJ'
SERPAPI_KEY = os.environ.get('SERPAPI_KEY')
ROOT = Path(__file__).resolve().parent.parent
PAPERS_TOML = ROOT / 'data' / 'papers.toml'
OUT_FILE = ROOT / 'data' / 'scholar.json'
SCHOLARLY_PAPERS_KEY = 'public' + 'ations'

def log(msg):
    print(msg, flush=True)

def norm(s):
    return re.sub(r'\W+', '', s.casefold())

def fetch_serpapi():
    log("  fetching via SerpApi...")
    # One author-list request supplies total and per-paper counts; never fetch details.
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

def write_stats(payload):
    """Replace the complete cache atomically so readers never see partial JSON."""
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=OUT_FILE.parent, suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            f.write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        temporary.replace(OUT_FILE)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def main():
    with PAPERS_TOML.open('rb') as f:
        papers = tomllib.load(f).get('papers', [])
    wanted = [p for p in papers if p.get('gs_id')]
    log(f"Tracking {len(wanted)} papers")
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

            previous = json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {}
            cached = previous.get('papers', {})
            # Retain unmatched counts, but drop works removed from the paper list.
            counts = {p['gs_id']: cached[p['gs_id']] for p in wanted if p['gs_id'] in cached}
            matched = 0
            for paper in wanted:
                gs_id = paper['gs_id']
                keys = [norm(paper['title']), norm(paper.get('gs_title', paper['title']))]
                title_key = next((key for key in keys if key in by_title), None)
                if title_key is not None:
                    counts[gs_id] = by_title[title_key]
                    matched += 1
                    log(f"  matched {gs_id}: {by_title[title_key]} citations")
                else:
                    log(f"  WARN no match for {gs_id} (title: {paper['title']!r})")

            write_stats({'author_id': AUTHOR_ID, 'total': total, 'papers': counts})
            log(f"Total: {total}, matched {matched}/{len(wanted)} papers")
            break
        except Exception as exc:
            last_error = exc
            log(f"Attempt {attempt} failed: {exc}")
            if attempt < 2:
                time.sleep(10)
    else:
        log(f"All attempts failed. Last error: {last_error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
