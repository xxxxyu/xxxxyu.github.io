"""Repository checks for multilingual content and generated-surface conventions."""

from __future__ import annotations

import subprocess
import sys
import json
import re
import tomllib
import unittest
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOT = REPO_ROOT / "content"


def load_front_matter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "+++":
        raise AssertionError(f"{path.relative_to(REPO_ROOT)} has no TOML front matter")
    closing = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "+++"),
        None,
    )
    if closing is None:
        raise AssertionError(f"{path.relative_to(REPO_ROOT)} has unterminated front matter")
    return tomllib.loads("\n".join(lines[1:closing]))


def markdown_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    _, body = text.split("+++", maxsplit=2)[1:]
    return body


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
        self.assertEqual(zh.get("title"), "李翔宇")
        self.assertIs(zh.get("generate_feeds"), True)
        self.assertIs(zh.get("build_search_index"), False)
        self.assertIn("atom.xml", zh.get("feed_filenames", []))
        self.assertIn("tags", {item.get("name") for item in zh.get("taxonomies", [])})

    def test_chinese_profile_and_about_bios_cover_the_english_source(self) -> None:
        home = CONTENT_ROOT / "_index.zh.md"
        about = CONTENT_ROOT / "about" / "_index.zh.md"
        home_body = markdown_body(home)
        about_body = markdown_body(about)
        self.assertNotIn("TODO", home_body)
        self.assertNotIn("TODO", about_body)
        self.assertIn("面向端侧智能的高效推理系统", home_body)
        self.assertIn("具身智能的模型系统协同设计", home_body)
        for term in ("刘云新", "曹婷", "FlexNN", "Vec-LUT", "OxyGen", "KV cache"):
            self.assertIn(term, about_body)
        self.assertEqual(load_front_matter(about).get("title"), "关于")

    def test_primary_section_labels_exist_in_both_languages(self) -> None:
        keys = {
            "position", "affiliation", "news", "selected_papers", "latest_posts",
            "see_all", "papers", "archive", "tags", "tagged", "experiences",
            "skills", "featured", "photography", "notes", "music",
        }
        english = self.config.get("translations", {})
        chinese = self.config.get("languages", {}).get("zh", {}).get("translations", {})
        self.assertTrue(keys.issubset(english))
        self.assertTrue(keys.issubset(chinese))
        for key in keys:
            self.assertTrue(english[key].strip(), key)
            self.assertTrue(chinese[key].strip(), key)

        about_template = (REPO_ROOT / "templates" / "about.html").read_text(encoding="utf-8")
        self.assertIn('data/experiences.zh.md', about_template)
        self.assertIn('data/skills.zh.md', about_template)
        experiences = (REPO_ROOT / "data" / "experiences.zh.md").read_text(encoding="utf-8")
        skills = (REPO_ROOT / "data" / "skills.zh.md").read_text(encoding="utf-8")
        for heading in ("### 奖项", "### 实习", "### 助教", "### 其他"):
            self.assertIn(heading, experiences)
        for heading in ("### 编程语言", "### 框架"):
            self.assertIn(heading, skills)

    def test_translation_catalog_has_no_unwired_keys(self) -> None:
        english = self.config.get("translations", {})
        chinese = self.config.get("languages", {}).get("zh", {}).get("translations", {})
        self.assertEqual(set(english), set(chinese))
        templates = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (REPO_ROOT / "templates").rglob("*.html")
        )
        for key in english:
            with self.subTest(key=key):
                self.assertIn(f'trans(key="{key}"', templates)

    def test_blog_surfaces_use_default_list_with_per_card_localization(self) -> None:
        index_template = (REPO_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        blog_template = (REPO_ROOT / "templates" / "blog.html").read_text(encoding="utf-8")
        series_template = (REPO_ROOT / "templates" / "series.html").read_text(encoding="utf-8")
        self.assertIn('lang=config.default_language', index_template)
        self.assertIn("localized_post_preview", index_template)
        self.assertIn('get_section(path="blog/_index.md", lang=config.default_language)', blog_template)
        self.assertIn("localized_post_preview", blog_template)
        self.assertNotIn("paginator.", blog_template)
        self.assertNotIn("paginate_by", (CONTENT_ROOT / "blog" / "_index.md").read_text(encoding="utf-8"))
        self.assertNotIn("paginate_by", (CONTENT_ROOT / "blog" / "_index.zh.md").read_text(encoding="utf-8"))
        self.assertIn("source_series.pages", series_template)
        self.assertIn("localized_series_item", series_template)

        post_template = (REPO_ROOT / "templates" / "blog-page.html").read_text(encoding="utf-8")
        macros = (REPO_ROOT / "templates" / "macros.html").read_text(encoding="utf-8")
        self.assertIn('get_section(path="blog/_index.md", lang=config.default_language)', post_template)
        self.assertIn("source_page.permalink", post_template)
        self.assertIn("localized_post_nav_link", post_template)
        self.assertIn("data-title-en", macros)
        self.assertIn("data-title-zh", macros)

    def test_language_preference_is_click_driven_and_keeps_untranslated_url(self) -> None:
        template = (REPO_ROOT / "templates" / "base.html").read_text(encoding="utf-8")
        blog_template = (REPO_ROOT / "templates" / "blog-page.html").read_text(encoding="utf-8")
        self.assertIn("localStorage.setItem('site-language', target)", template)
        self.assertIn("switcher.href = hasExactTranslation ? alternate.href : window.location.href", template)
        self.assertNotIn("opens the Chinese home page", template)
        for key in ("toc", "contents", "previous", "next", "post_navigation", "updated"):
            self.assertIn(f'{key}:', template)
        self.assertIn('data-i18n-text="previous"', blog_template)
        self.assertIn('data-i18n-text="next"', blog_template)
        search_script = (REPO_ROOT / "static" / "js" / "search.js").read_text(encoding="utf-8")
        self.assertIn("docsLanguage === lang", search_script)

    def test_generated_surfaces_include_multilingual_alternates(self) -> None:
        sitemap = (REPO_ROOT / "templates" / "sitemap.xml").read_text(encoding="utf-8")
        taxonomy_list = (REPO_ROOT / "templates" / "taxonomy_list.html").read_text(encoding="utf-8")
        taxonomy_single = (REPO_ROOT / "templates" / "taxonomy_single.html").read_text(encoding="utf-8")
        self.assertIn('xmlns:xhtml="http://www.w3.org/1999/xhtml"', sitemap)
        self.assertIn('hreflang="x-default"', sitemap)
        self.assertIn('hreflang="x-default"', taxonomy_list)
        self.assertIn('hreflang="x-default"', taxonomy_single)
        self.assertIn('get_taxonomy(kind="tags", lang=config.default_language)', taxonomy_single)

    def test_ai_translations_declare_and_link_their_source(self) -> None:
        declared = []
        for page in (CONTENT_ROOT / "blog").rglob("index*.md"):
            metadata = load_front_matter(page)
            source_language = metadata.get("extra", {}).get("ai_translation_source")
            if source_language:
                declared.append(page)
                self.assertIn(source_language, {"en", "zh"})
                page_language = "zh" if page.name.endswith(".zh.md") else "en"
                self.assertNotEqual(source_language, page_language)
                source_name = "index.zh.md" if source_language == "zh" else "index.md"
                self.assertTrue(page.with_name(source_name).is_file())
        self.assertTrue(declared, "At least one AI translation must exercise the disclosure")
        for english in (CONTENT_ROOT / "blog").rglob("index.md"):
            chinese = english.with_name("index.zh.md")
            declarations = [
                load_front_matter(path).get("extra", {}).get("ai_translation_source")
                for path in (english, chinese)
            ]
            with self.subTest(pair=english.parent.relative_to(CONTENT_ROOT)):
                self.assertEqual(sum(bool(value) for value in declarations), 1)
        template = (REPO_ROOT / "templates" / "blog-page.html").read_text(encoding="utf-8")
        self.assertIn("ai_translation_source", template)
        self.assertIn('trans(key="ai_translated_from"', template)
        self.assertIn('hreflang="{{ ai_translation_source_lang }}"', template)
        self.assertIn('data-set-site-language="{{ ai_translation_source_lang }}"', template)
        base_template = (REPO_ROOT / "templates" / "base.html").read_text(encoding="utf-8")
        self.assertIn("localStorage.setItem('site-language', target)", base_template)
        self.assertIn("[data-set-site-language]", base_template)

    def test_committed_chinese_search_index_is_current(self) -> None:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "build_zh_search_index.py"), "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        index = json.loads((REPO_ROOT / "static" / "search_index.zh.json").read_text(encoding="utf-8"))
        documents = {document["path"]: document for document in index["documents"]}
        default_blog = [
            path for path in (CONTENT_ROOT / "blog").rglob("*.md")
            if not path.name.startswith("_index")
            and not path.name.endswith(".zh.md")
            and load_front_matter(path).get("draft") is not True
        ]
        indexed_post_paths = {
            path for path in documents
            if path.startswith("/blog/") or path.startswith("/zh/blog/")
        }
        self.assertGreaterEqual(len(indexed_post_paths), len(default_blog))
        for source in default_blog:
            translated = source.with_name("index.zh.md")
            translated_meta = load_front_matter(translated)
            expected = translated_meta.get("path")
            if expected:
                expected_path = f"/{expected.strip('/')}/"
            else:
                expected_path = "/zh/" + "/".join(translated.relative_to(CONTENT_ROOT).parent.parts) + "/"
            source_meta = load_front_matter(source)
            source_path = source_meta.get("path") or "/".join(source.relative_to(CONTENT_ROOT).parent.parts)
            with self.subTest(source=source.relative_to(REPO_ROOT)):
                self.assertIn(expected_path, documents)
                self.assertNotIn(f"/{source_path.strip('/')}/", documents)

    def test_every_chinese_file_has_a_colocated_english_source(self) -> None:
        self.assertTrue(self.translations, "No Chinese translations were found")
        for translated in self.translations:
            source = translated.with_name(translated.name.removesuffix(".zh.md") + ".md")
            with self.subTest(translation=translated.relative_to(REPO_ROOT)):
                self.assertTrue(source.is_file(), f"Missing paired source {source}")

    def test_every_blog_post_has_an_english_chinese_pair(self) -> None:
        default_posts = sorted((CONTENT_ROOT / "blog").rglob("index.md"))
        self.assertTrue(default_posts)
        for source in default_posts:
            with self.subTest(source=source.relative_to(REPO_ROOT)):
                self.assertTrue(source.with_name("index.zh.md").is_file())

    def test_first_post_pair_uses_stable_language_prefixed_urls(self) -> None:
        translated_pages = [path for path in self.translations if not path.name.startswith("_index")]
        self.assertTrue(translated_pages, "The MVP needs at least one translated post")
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

    def test_post_titles_descriptions_and_code_blocks_are_safe_for_the_layout(self) -> None:
        for translated in self.translations:
            if translated.name.startswith("_index"):
                continue
            source = translated.with_name(translated.name.removesuffix(".zh.md") + ".md")
            metadata = load_front_matter(translated)
            with self.subTest(translation=translated.relative_to(REPO_ROOT)):
                self.assertLessEqual(len(metadata.get("title", "")), 70)
                self.assertLessEqual(len(metadata.get("description", "")), 160)
                source_structure = markdown_structure(markdown_body(source))
                translated_structure = markdown_structure(markdown_body(translated))
                self.assertEqual(translated_structure, source_structure)


if __name__ == "__main__":
    unittest.main()
