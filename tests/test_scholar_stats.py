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
    def test_all_papers_are_tracked_with_unique_ids(self):
        papers = tomllib.loads(stats.PAPERS_TOML.read_text())['papers']
        ids = [p['gs_id'] for p in papers]
        self.assertEqual(len(ids), len(set(ids)))

    def test_one_author_request_updates_all_papers_including_below_threshold(self):
        papers = tomllib.loads(stats.PAPERS_TOML.read_text())['papers']
        counts = [9, 10, 11] + list(range(len(papers) - 3))
        payload = {
            'cited_by': {'table': [{'citations': {'all': 749}}]},
            'articles': [
                {'title': p.get('gs_title', p['title']), 'cited_by': {'value': n}}
                for p, n in zip(papers, counts)
            ],
        }
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(stats, 'OUT_FILE', Path(directory) / 'scholar.json'), \
             patch.object(stats, 'SERPAPI_KEY', 'test-key'), \
             patch.object(stats.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(payload).encode())) as request:
            self.assertIsNone(stats.main())
            request.assert_called_once()
            query = parse_qs(urlsplit(request.call_args.args[0]).query)
            self.assertEqual(query['engine'], ['google_scholar_author'])
            self.assertEqual(query['num'], ['100'])
            self.assertNotIn('citation_id', query)
            cache = json.loads(stats.OUT_FILE.read_text())
            self.assertEqual(cache['total'], 749)
            self.assertEqual(list(Path(directory).iterdir()), [stats.OUT_FILE])
            for paper, count in zip(papers, counts):
                self.assertEqual(cache['papers'][paper['gs_id']], count)

    def test_punctuation_and_spacing_differences_match(self):
        self.assertEqual(stats.norm('Embodied. cpp: A Runtime'), stats.norm('Embodied.cpp: A Runtime'))

    def test_fetch_failure_preserves_previous_counts(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(stats, 'OUT_FILE', Path(directory) / 'scholar.json'), \
             patch.object(stats, 'SERPAPI_KEY', 'test-key'), \
             patch.object(stats.urllib.request, 'urlopen', side_effect=OSError('unavailable')) as request, \
             patch.object(stats.time, 'sleep'):
            stats.write_stats({'author_id': stats.AUTHOR_ID, 'total': 749, 'papers': {'flexnn': 77}})
            original = stats.OUT_FILE.read_bytes()
            self.assertEqual(stats.main(), 1)
            self.assertEqual(request.call_count, 2)
            self.assertEqual(stats.OUT_FILE.read_bytes(), original)

    def test_partial_response_retains_unmatched_counts_and_drops_removed_papers(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(stats, 'OUT_FILE', Path(directory) / 'scholar.json'), \
             patch.object(stats, 'SERPAPI_KEY', 'test-key'), \
             patch.object(stats, 'fetch_serpapi', return_value=(750, {stats.norm('FlexNN: Efficient and Adaptive DNN Inference on Memory-Constrained Edge Devices'): 78})):
            stats.write_stats({'author_id': stats.AUTHOR_ID, 'total': 749,
                               'papers': {'flexnn': 77, 'zetta': 17, 'removed-paper': 12}})
            self.assertIsNone(stats.main())
            cache = json.loads(stats.OUT_FILE.read_text())
            self.assertEqual(cache['total'], 750)
            self.assertEqual(cache['papers'], {'flexnn': 78, 'zetta': 17})

    def test_failed_atomic_replace_preserves_cache(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(stats, 'OUT_FILE', Path(directory) / 'scholar.json'):
            stats.write_stats({'total': 749, 'papers': {'flexnn': 77}})
            original = stats.OUT_FILE.read_bytes()
            with patch.object(Path, 'replace', side_effect=OSError('write failed')):
                with self.assertRaises(OSError):
                    stats.write_stats({'total': 750, 'papers': {'flexnn': 78}})
            self.assertEqual(stats.OUT_FILE.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [stats.OUT_FILE])


if __name__ == '__main__':
    unittest.main()
