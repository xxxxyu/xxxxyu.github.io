#!/usr/bin/env python3
"""Cache the complete Scholar author listing, keyed by Scholar citation ID."""
from datetime import datetime, timezone
import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

AUTHOR_ID = 'IjoWeIMAAAAJ'
ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / 'data' / 'scholar.json'


def report(message):
    print(message, flush=True)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        try:
            with open(summary, 'a', encoding='utf-8') as f:
                f.write(message + '\n\n')
        except OSError:
            print('Could not append Actions Summary. Refresh details are available in this log.', flush=True)


def request_serpapi(endpoint, params, api_key):
    query = urllib.parse.urlencode({**params, 'api_key': api_key})
    with urllib.request.urlopen(f'https://serpapi.com/{endpoint}?{query}', timeout=30) as r:
        payload = json.loads(r.read())
    if not isinstance(payload, dict):
        raise RuntimeError('SerpApi returned a response without a JSON object')
    if payload.get('error'):
        raise RuntimeError(f"SerpApi error {payload['error']}")
    return payload


def check_account(api_key):
    account = request_serpapi('account.json', {}, api_key)
    remaining = account.get('total_searches_left', account.get('plan_searches_left'))
    if remaining is None:
        raise RuntimeError('SerpApi account response is missing the remaining quota')
    # The account response also contains credentials and account identity.
    # Only these usage fields reach logs, never the full response.
    report(
        f"SerpApi {account.get('plan_name', 'plan')}. "
        f"Used {account.get('this_month_usage', 'unknown')} / "
        f"{account.get('searches_per_month', 'unknown')} searches. "
        f"Remaining before this search {remaining}. "
        f"Next renewal {account.get('plan_renewal_date') or 'unknown'}."
    )
    if int(remaining) <= 0:
        raise RuntimeError('SerpApi search quota is exhausted')


def citation_count(article):
    cited = article.get('cited_by', {})
    count = int((cited.get('value') if isinstance(cited, dict) else cited) or 0)
    if count < 0:
        raise ValueError('Scholar returned a negative citation count')
    return count


def fetch_serpapi(api_key):
    payload = request_serpapi('search.json', {
        'engine': 'google_scholar_author',
        'author_id': AUTHOR_ID,
        'hl': 'en',
        'num': '100',
    }, api_key)
    return parse_snapshot(payload)


def parse_snapshot(payload):
    """Validate a complete author listing and index its public article data."""
    if payload.get('serpapi_pagination', {}).get('next'):
        raise RuntimeError('Scholar listing exceeds one page of 100 articles')

    total = None
    for row in payload.get('cited_by', {}).get('table', []):
        cits = row.get('citations', {})
        if cits.get('all') is not None:
            total = int(cits['all'])
            break

    articles = payload.get('articles')
    if total is None or total < 0 or not isinstance(articles, list):
        raise RuntimeError('Scholar response is missing author totals or the article list')
    if total and not articles:
        raise RuntimeError('Scholar response has citations but an empty article list')
    articles_by_id = {}
    for article in articles:
        if not isinstance(article, dict):
            raise RuntimeError('Scholar response contains an invalid article')
        citation_id = article.get('citation_id')
        if not isinstance(citation_id, str) or not citation_id or not article.get('title'):
            raise RuntimeError('Scholar response contains an article without a title or citation ID')
        if citation_id in articles_by_id:
            raise RuntimeError(f'Scholar response repeats article {citation_id}')
        cited_by = article.get('cited_by')
        if not isinstance(cited_by, dict):
            cited_by = {}
        articles_by_id[citation_id] = {
            **article,
            'cited_by': {**cited_by, 'value': citation_count(article)},
        }

    # Preserve every article and all author statistics, indexed by the source ID.
    # Fetch timestamps and request metadata belong in Actions Summary.
    return {
        'author_id': AUTHOR_ID,
        'author': payload.get('author', {}),
        'total': total,
        'cited_by': payload['cited_by'],
        'articles': articles_by_id,
    }


def write_stats(payload):
    """Replace the complete cache atomically so readers never see partial JSON."""
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + '\n'
    if OUT_FILE.exists() and OUT_FILE.read_text(encoding='utf-8') == serialized:
        return False
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=OUT_FILE.parent, suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            f.write(serialized)
        temporary.replace(OUT_FILE)
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    api_key = os.environ.get('SERPAPI_KEY', '').strip()
    report('## Google Scholar refresh')
    report(f"Checked at {datetime.now(timezone.utc).isoformat(timespec='seconds')}.")
    try:
        if not api_key:
            raise RuntimeError('Set SERPAPI_KEY to refresh Scholar data')
        check_account(api_key)
        snapshot = fetch_serpapi(api_key)
        changed = write_stats(snapshot)
        report(
            f"Retrieved {len(snapshot['articles'])} articles and {snapshot['total']} total citations. "
            f"Cache {'updated' if changed else 'unchanged'}."
        )
        return 0
    except Exception as exc:
        message = str(exc).replace(api_key, '[redacted]') if api_key else str(exc)
        report(f'Scholar refresh failed. {message}. Existing cache retained.')
        return 1


if __name__ == "__main__":
    sys.exit(main())
