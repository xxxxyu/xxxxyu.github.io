import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from scripts import update_scholar_stats as stats


class ScholarStatsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / 'scholar.json'
        self.summary = Path(directory.name) / 'summary.md'
        self.log = io.StringIO()
        self.enterContext(patch.object(stats, 'OUT_FILE', self.output))
        self.enterContext(patch.dict('os.environ', {
            'SERPAPI_KEY': 'test-key', 'GITHUB_STEP_SUMMARY': str(self.summary),
        }))
        self.enterContext(contextlib.redirect_stdout(self.log))
        self.account = {
            'api_key': 'test-key',
            'account_email': 'private@example.com',
            'plan_name': 'Free Plan',
            'searches_per_month': 250,
            'this_month_usage': 3,
            'total_searches_left': 247,
            'plan_renewal_date': '2026-12-01',
        }
        self.payload = {
            'author': {'name': 'Xiangyu Li'},
            'cited_by': {'table': [{'citations': {'all': 749}}]},
            'articles': [
                {
                    'citation_id': f'{stats.AUTHOR_ID}:paper-{index}',
                    'title': 'An identical title',
                    'authors': 'X Li, Y Zhang',
                    'publication': 'A venue',
                    'year': '2026',
                    'cited_by': {'value': count, 'link': f'https://scholar.google.com/scholar?cites={index}'},
                }
                for index, count in enumerate((9, 10, 11))
            ],
        }

    def responses(self, *payloads):
        replies = [payload if isinstance(payload, Exception)
                   else io.BytesIO(json.dumps(payload).encode()) for payload in payloads]
        return patch.object(stats.urllib.request, 'urlopen', side_effect=replies)

    def seed_cache(self):
        stats.write_stats({'author_id': stats.AUTHOR_ID, 'total': 748,
                           'articles': {f'{stats.AUTHOR_ID}:removed': {'cited_by': {'value': 77}}}})
        return self.output.read_bytes()

    def test_site_publications_and_cache_use_valid_unique_scholar_ids(self):
        papers = tomllib.loads((stats.ROOT / 'data/papers.toml').read_text())['papers']
        cache = json.loads((stats.ROOT / 'data/scholar.json').read_text())
        ids = [paper['citation_id'] for paper in papers if paper.get('citation_id')]
        self.assertEqual(len(ids), len(set(ids)))
        for citation_id in ids:
            self.assertRegex(citation_id, rf'^{stats.AUTHOR_ID}:[A-Za-z0-9_-]+$')
        for citation_id, article in cache['articles'].items():
            self.assertEqual(citation_id, article['citation_id'])
            self.assertGreaterEqual(article['cited_by']['value'], 0)

    def test_refresh_checks_quota_once_and_saves_the_complete_id_indexed_snapshot(self):
        self.seed_cache()
        self.payload['search_metadata'] = {'created_at': 'volatile timestamp'}
        self.payload['search_parameters'] = {'api_key': 'test-key'}
        with self.responses(self.account, self.payload) as request:
            self.assertEqual(stats.main(), 0)
        self.assertEqual(request.call_count, 2)
        urls = [urlsplit(call.args[0]) for call in request.call_args_list]
        self.assertEqual([url.path for url in urls], ['/account.json', '/search.json'])
        query = parse_qs(urls[1].query)
        self.assertEqual(parse_qs(urls[0].query)['api_key'], ['test-key'])
        self.assertEqual(query['api_key'], ['test-key'])
        self.assertEqual(query['engine'], ['google_scholar_author'])
        self.assertEqual(query['num'], ['100'])
        self.assertNotIn('citation_id', query)
        self.assertNotIn('no_cache', query)
        self.assertEqual(json.loads(self.output.read_text()), {
            'author_id': stats.AUTHOR_ID,
            'author': self.payload['author'],
            'total': 749,
            'cited_by': self.payload['cited_by'],
            'articles': {article['citation_id']: article for article in self.payload['articles']},
        })
        for output in (self.output.read_text(), self.summary.read_text(), self.log.getvalue()):
            self.assertNotIn('test-key', output)
            self.assertNotIn('private@example.com', output)
        self.assertIn('Remaining before this search 247', self.summary.read_text())

    def test_failed_refresh_preserves_cache_and_limits_requests(self):
        original = self.seed_cache()
        cases = (
            ('missing key', '', [], 0),
            ('exhausted quota', 'test-key', [{**self.account, 'total_searches_left': 0}], 1),
            ('unknown quota', 'test-key', [{**self.account, 'total_searches_left': None}], 1),
            ('account unavailable', 'test-key', [OSError('unavailable')], 1),
            ('search unavailable', 'test-key', [self.account, OSError('unavailable')], 2),
            ('API error', 'test-key', [self.account, {'error': 'request failed using test-key'}], 2),
            ('invalid JSON shape', 'test-key', [self.account, []], 2),
            ('incomplete results', 'test-key', [self.account, {
                **self.payload, 'serpapi_pagination': {'next': 'https://serpapi.com/search.json?start=100'},
            }], 2),
        )
        for label, key, responses, count in cases:
            with self.subTest(failure=label), \
                 patch.dict('os.environ', {'SERPAPI_KEY': key}), \
                 self.responses(*responses) as request:
                self.assertEqual(stats.main(), 1)
                self.assertEqual(request.call_count, count)
                self.assertEqual(self.output.read_bytes(), original)
        self.assertNotIn('test-key', self.summary.read_text() + self.log.getvalue())

    def test_invalid_author_listings_are_rejected(self):
        duplicate = copy.deepcopy(self.payload)
        duplicate['articles'].append(duplicate['articles'][0])
        negative = copy.deepcopy(self.payload)
        negative['articles'][0]['cited_by']['value'] = -1
        cases = (
            duplicate, negative,
            {**self.payload, 'articles': []},
            {**self.payload, 'articles': None},
            {**self.payload, 'articles': ['invalid article']},
            {**self.payload, 'articles': [{'title': 'Missing citation ID'}]},
            {**self.payload, 'cited_by': {}},
        )
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises((ValueError, RuntimeError)):
                stats.parse_snapshot(payload)

    def test_citation_values_are_normalized_and_source_links_are_preserved(self):
        values = [None, 9, {'value': '11', 'link': 'https://scholar.google.com/scholar?cites=2'}]
        for article, value in zip(self.payload['articles'], values):
            article['cited_by'] = value
        articles = stats.parse_snapshot(self.payload)['articles']
        self.assertEqual([articles[a['citation_id']]['cited_by']['value'] for a in self.payload['articles']], [0, 9, 11])
        self.assertIn('link', articles[self.payload['articles'][2]['citation_id']]['cited_by'])

    def test_zero_total_and_zero_citation_counts_are_valid(self):
        self.payload['cited_by']['table'][0]['citations']['all'] = 0
        for article in self.payload['articles']:
            article['cited_by']['value'] = 0
        snapshot = stats.parse_snapshot(self.payload)
        self.assertEqual(snapshot['total'], 0)
        self.assertTrue(all(article['cited_by']['value'] == 0 for article in snapshot['articles'].values()))

    def test_repeated_refresh_avoids_writes_even_when_the_summary_is_unavailable(self):
        with self.responses(self.account, self.payload):
            self.assertEqual(stats.main(), 0)
        original_mtime = self.output.stat().st_mtime_ns
        self.payload['articles'].reverse()
        self.payload['search_metadata'] = {'created_at': 'a new fetch time'}
        with self.responses(self.account, self.payload), \
             patch.dict('os.environ', {'GITHUB_STEP_SUMMARY': str(self.output.parent)}), \
             patch.object(Path, 'replace') as replace:
            self.assertEqual(stats.main(), 0)
        replace.assert_not_called()
        self.assertEqual(self.output.stat().st_mtime_ns, original_mtime)

    def test_failed_atomic_replace_preserves_cache_and_removes_temp_file(self):
        original = self.seed_cache()
        with patch.object(Path, 'replace', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):
                stats.write_stats({'total': 750, 'articles': {}})
        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])


if __name__ == '__main__':
    unittest.main()
