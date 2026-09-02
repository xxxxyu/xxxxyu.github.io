import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class FrontendConventionTests(unittest.TestCase):
    def test_footnote_layout_and_script_use_the_rendered_definition_marker(self) -> None:
        template = (REPO_ROOT / "templates" / "blog-page.html").read_text(encoding="utf-8")

        marker = "page.content is containing('class=\"footnote-definition\"')"
        self.assertEqual(template.count(marker), 3)
        self.assertIn("{% if has_footnotes %}\n<aside", template)
        self.assertIn("path='js/blog-footnotes.js'", template)

    def test_outside_image_priority_is_explicit_and_unique_by_construction(self) -> None:
        macros = (REPO_ROOT / "templates" / "macros.html").read_text(encoding="utf-8")
        template = (REPO_ROOT / "templates" / "outside.html").read_text(encoding="utf-8")

        self.assertIn("macro outside_card(page, variant, priority)", macros)
        self.assertIn("{% if priority %}loading=\"eager\" fetchpriority=\"high\"", macros)
        self.assertIn("priority=not has_priority_cover", template)
        self.assertIn("{% set_global has_priority_cover = true %}", template)

    def test_runtime_behaviors_live_in_static_scripts(self) -> None:
        base = (REPO_ROOT / "templates" / "base.html").read_text(encoding="utf-8")
        language = (REPO_ROOT / "static" / "js" / "site-language.js").read_text(encoding="utf-8")
        theme = (REPO_ROOT / "static" / "js" / "theme.js").read_text(encoding="utf-8")

        self.assertIn("path='js/site-language.js'", base)
        self.assertIn("path='js/theme.js'", base)
        self.assertNotIn("function storedLanguage()", base)
        self.assertIn("function storedLanguage()", language)
        self.assertIn("var cycle = ['light', 'dark', 'system'];", theme)

    def test_blog_styles_are_split_by_surface(self) -> None:
        entrypoint = (REPO_ROOT / "sass" / "style.scss").read_text(encoding="utf-8")

        for partial in ("blog", "blog-series", "blog-index"):
            self.assertIn(f'@import "{partial}";', entrypoint)
        self.assertTrue((REPO_ROOT / "sass" / "_blog-series.scss").is_file())
        self.assertTrue((REPO_ROOT / "sass" / "_blog-index.scss").is_file())

    def test_footer_uses_the_shared_bullet_separator(self) -> None:
        template = (REPO_ROOT / "templates" / "base.html").read_text(encoding="utf-8")

        self.assertNotIn("&middot;", template)
        self.assertEqual(template.count('class="bullet-separator"'), 2)

    def test_paper_card_separates_press_links_from_unstyled_notes(self) -> None:
        macros = (REPO_ROOT / "templates" / "macros.html").read_text(encoding="utf-8")

        self.assertIn("for link in pub.press_links | default(value=[])", macros)
        self.assertNotIn("<em>{{ notes", macros)

    def test_ci_uses_a_pinned_zola_release(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")

        self.assertRegex(workflow, r'ZOLA_VERSION:\s+"0\.22\.1"')
        self.assertNotIn("/releases/latest", workflow)


if __name__ == "__main__":
    unittest.main()
