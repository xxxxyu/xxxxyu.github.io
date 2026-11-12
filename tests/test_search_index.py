"""Search generation works from source content, without a committed JSON file."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_zh_search_index as search


class SearchIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "config.toml").write_text('base_url = "https://example.test/"\n')
        self.output = self.root / "static/search_index.zh.json"

    def page(self, path: str, metadata: str = '', body: str = 'Text') -> Path:
        source = self.root / "content" / path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f'+++\n{metadata}\n+++\n\n{body}\n', encoding="utf-8")
        return source

    def test_translations_fallbacks_and_search_opt_outs(self) -> None:
        self.page("post/index.md", 'title = "English"', "Original")
        self.page("post/index.zh.md", 'title = "中文"\ndescription = "简介"', "中文 **正文**")
        self.page("fallback.md", 'title = "Fallback"')
        for flag in ('draft = true', 'in_search_index = false'):
            name = flag.split()[0]
            self.page(f"excluded-{name}.md", flag)
            self.page(f"excluded-{name}.zh.md", 'title = "Also excluded"')
            self.page(f"fallback-{name}.md", 'title = "Published English"')
            self.page(f"fallback-{name}.zh.md", flag)

        documents = {item['path']: item for item in search.build_index(self.root)['documents']}
        self.assertEqual(set(documents), {
            "/zh/post/", "/fallback/", "/fallback-draft/", "/fallback-in_search_index/",
        })
        translated = documents["/zh/post/"]
        self.assertEqual(translated['id'], "https://example.test/zh/post/")
        self.assertEqual(translated['title'], "中文")
        self.assertEqual(translated['description'], "简介")
        self.assertEqual(translated['body'], "中文 正文")
        self.assertFalse(self.output.exists(), "Building in memory must not write an index")

    def test_section_and_explicit_paths(self) -> None:
        self.page("_index.md")
        self.page("_index.zh.md")
        self.page("about/_index.md")
        self.page("about/_index.zh.md")
        self.page("post.md")
        self.page("post.zh.md", 'path = "/zh/custom-location/"')
        paths = {item['path'] for item in search.build_index(self.root)['documents']}
        self.assertEqual(paths, {"/zh/", "/zh/about/", "/zh/custom-location/"})

    def test_cold_generation_no_op_writes_and_content_edits(self) -> None:
        self.page("post.md")
        source = self.page("post.zh.md", 'title = "标题"', "甲乙")
        self.assertTrue(search.ensure_index(self.root))
        original = self.output.read_bytes()
        os.utime(self.output, ns=(1_000_000_000, 1_000_000_000))
        self.assertFalse(search.ensure_index(self.root))
        self.assertEqual(self.output.stat().st_mtime_ns, 1_000_000_000)

        # Same glyphs, different text: search still needs regeneration.
        source.write_text(source.read_text().replace("甲乙", "乙甲"), encoding="utf-8")
        self.assertTrue(search.ensure_index(self.root))
        self.assertNotEqual(self.output.read_bytes(), original)
        self.assertEqual(json.loads(self.output.read_text())['documents'][0]['body'], "乙甲")
        source.unlink()
        self.assertTrue(search.ensure_index(self.root))
        self.assertEqual(json.loads(self.output.read_text())['documents'][0]['path'], "/post/")
        (self.root / "content/post.md").unlink()
        self.assertTrue(search.ensure_index(self.root))
        self.assertEqual(json.loads(self.output.read_text())['documents'], [])

    def test_invalid_content_preserves_previous_index(self) -> None:
        source = self.page("post.md")
        search.ensure_index(self.root)
        original = self.output.read_bytes()
        source.write_text("+++\ntitle = \"unfinished\n+++\nText")
        with self.assertRaises(ValueError):
            search.ensure_index(self.root)
        self.assertEqual(self.output.read_bytes(), original)

    def test_failed_replace_preserves_previous_index_and_removes_temporary_file(self) -> None:
        source = self.page("post.md")
        search.ensure_index(self.root)
        original = self.output.read_bytes()
        source.write_text(source.read_text().replace("Text", "Updated text"))
        with mock.patch.object(Path, "replace", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                search.ensure_index(self.root)
        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])


if __name__ == "__main__":
    unittest.main()
