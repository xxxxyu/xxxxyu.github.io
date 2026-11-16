"""Repository checks for multilingual content and generated-surface conventions."""

from __future__ import annotations

import re
import tomllib
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.build_zh_search_index import split_content


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOT = REPO_ROOT / "content"


def load_front_matter(path: Path) -> dict[str, Any]:
    return split_content(path)[0]


def markdown_body(path: Path) -> str:
    return split_content(path)[1]


def markdown_structure(body: str) -> dict[str, Counter[str] | int]:
    links = re.findall(r"]\(([^)\n]+)\)", body)
    return {
        "external_links": Counter(link for link in links if link.startswith(("http://", "https://"))),
        "site_links": Counter(
            link.removeprefix("/zh")
            for link in links
            if link.startswith("/")
        ),
        "fragment_link_count": sum(link.startswith("#") for link in links),
        "footnotes": Counter(re.findall(r"\[\^[^\]]+\]", body)),
        "shortcode_sources": Counter(re.findall(r'\{\{\s*\w+\([^\n]*?src="([^"]+)"', body)),
        "headings": Counter(re.findall(r"(?m)^#{2,4}\s+", body)),
        "code_fences": body.count("```"),
    }


class MultilingualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = tomllib.loads((REPO_ROOT / "config.toml").read_text(encoding="utf-8"))
        cls.translations = sorted(CONTENT_ROOT.rglob("*.zh.md"))

    def test_language_configuration_enables_all_generated_surfaces(self) -> None:
        self.assertEqual(self.config.get("default_language"), "en")
        zh = self.config.get("languages", {}).get("zh", {})
        self.assertIs(zh.get("generate_feeds"), True)
        self.assertIs(zh.get("build_search_index"), False)
        self.assertIn("atom.xml", zh.get("feed_filenames", []))
        self.assertIn("tags", {item.get("name") for item in zh.get("taxonomies", [])})

    def test_chinese_profile_and_about_bios_match_source_structure(self) -> None:
        pairs = (
            (CONTENT_ROOT / "_index.md", CONTENT_ROOT / "_index.zh.md"),
            (CONTENT_ROOT / "about" / "_index.md", CONTENT_ROOT / "about" / "_index.zh.md"),
        )
        for source, translation in pairs:
            with self.subTest(source=source.relative_to(REPO_ROOT)):
                source_body = markdown_body(source)
                translation_body = markdown_body(translation)
                self.assertEqual(markdown_structure(translation_body), markdown_structure(source_body))

    def test_about_data_translations_preserve_structure(self) -> None:
        for section in ("awards", "experiences"):
            with self.subTest(section=section):
                english = REPO_ROOT / "data" / f"{section}.md"
                chinese = REPO_ROOT / "data" / f"{section}.zh.md"
                self.assertTrue(english.read_text(encoding="utf-8").strip())
                self.assertTrue(chinese.read_text(encoding="utf-8").strip())
                self.assertEqual(
                    markdown_structure(chinese.read_text(encoding="utf-8")),
                    markdown_structure(english.read_text(encoding="utf-8")),
                )

    def test_translation_catalogs_have_matching_keys(self) -> None:
        english = self.config.get("translations", {})
        chinese = self.config.get("languages", {}).get("zh", {}).get("translations", {})
        self.assertEqual(set(english), set(chinese))

    def test_ai_translation_metadata_references_the_original_page(self) -> None:
        for page in (CONTENT_ROOT / "blog").rglob("index*.md"):
            metadata = load_front_matter(page)
            source_language = metadata.get("extra", {}).get("ai_translation_source")
            if source_language:
                ai_translation = metadata["extra"]
                self.assertIn(source_language, {"en", "zh"})
                for field in ("ai_translation_harness", "ai_translation_model"):
                    with self.subTest(page=page.relative_to(REPO_ROOT), field=field):
                        self.assertIsInstance(ai_translation.get(field), str)
                        self.assertTrue(ai_translation[field].strip())
                self.assertIsInstance(ai_translation.get("ai_translation_effort"), str)
                page_language = "zh" if page.name.endswith(".zh.md") else "en"
                self.assertNotEqual(source_language, page_language)
                source_name = "index.zh.md" if source_language == "zh" else "index.md"
                self.assertTrue(page.with_name(source_name).is_file())
        for chinese in (CONTENT_ROOT / "blog").rglob("index.zh.md"):
            english = chinese.with_name("index.md")
            declarations = [
                load_front_matter(path).get("extra", {}).get("ai_translation_source")
                for path in (english, chinese)
            ]
            with self.subTest(pair=english.parent.relative_to(CONTENT_ROOT)):
                self.assertLessEqual(sum(bool(value) for value in declarations), 1)

    def test_every_chinese_file_has_a_colocated_english_source(self) -> None:
        self.assertTrue(self.translations, "No Chinese translations were found")
        for translated in self.translations:
            source = translated.with_name(translated.name.removesuffix(".zh.md") + ".md")
            with self.subTest(translation=translated.relative_to(REPO_ROOT)):
                self.assertTrue(source.is_file(), f"Missing paired source {source}")

    def test_explicit_translation_paths_use_the_language_prefix(self) -> None:
        translated_pages = [path for path in self.translations if not path.name.startswith("_index")]
        for translated in translated_pages:
            source = translated.with_name(translated.name.removesuffix(".zh.md") + ".md")
            source_meta = load_front_matter(source)
            translated_meta = load_front_matter(translated)
            with self.subTest(translation=translated.relative_to(REPO_ROOT)):
                if source_meta.get("path"):
                    self.assertEqual(translated_meta.get("path"), f"zh/{source_meta['path']}")
                else:
                    self.assertIsNone(translated_meta.get("path"))

    def test_key_metadata_is_synchronized_within_post_pairs(self) -> None:
        keys = ("date", "updated", "weight")
        for translated in self.translations:
            if translated.name.startswith("_index"):
                continue
            source = translated.with_name(translated.name.removesuffix(".zh.md") + ".md")
            source_meta = load_front_matter(source)
            translated_meta = load_front_matter(translated)
            with self.subTest(translation=translated.relative_to(REPO_ROOT)):
                for key in keys:
                    self.assertEqual(translated_meta.get(key), source_meta.get(key), key)
                self.assertEqual(translated_meta.get("taxonomies", {}).get("tags"), source_meta.get("taxonomies", {}).get("tags"))
                self.assertEqual(translated_meta.get("extra", {}).get("series"), source_meta.get("extra", {}).get("series"))

    def test_translated_series_membership_and_section_convention(self) -> None:
        for translated in self.translations:
            if translated.name.startswith("_index"):
                continue
            metadata = load_front_matter(translated)
            pointer = metadata.get("extra", {}).get("series")
            if not pointer:
                continue
            translated_section = (CONTENT_ROOT / pointer).with_name("_index.zh.md")
            section_meta = load_front_matter(translated_section)
            with self.subTest(translation=translated.relative_to(REPO_ROOT)):
                self.assertTrue(translated.is_relative_to(translated_section.parent))
                self.assertEqual(section_meta.get("template"), "series.html")
                self.assertEqual(section_meta.get("page_template"), "blog-page.html")
                self.assertEqual(section_meta.get("sort_by"), "weight")
                self.assertIs(section_meta.get("transparent"), True)
                self.assertIs(section_meta.get("extra", {}).get("series"), True)

    def test_post_translations_preserve_links_code_and_structure(self) -> None:
        for translated in self.translations:
            if translated.name.startswith("_index"):
                continue
            source = translated.with_name(translated.name.removesuffix(".zh.md") + ".md")
            with self.subTest(translation=translated.relative_to(REPO_ROOT)):
                source_structure = markdown_structure(markdown_body(source))
                translated_structure = markdown_structure(markdown_body(translated))
                self.assertEqual(translated_structure, source_structure)


if __name__ == "__main__":
    unittest.main()
