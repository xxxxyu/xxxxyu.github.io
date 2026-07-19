"""Repository checks for the post-series content convention."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOT = REPO_ROOT / "content"
BLOG_ROOT = CONTENT_ROOT / "blog"


def load_front_matter(path: Path) -> dict[str, Any]:
    """Load TOML front matter from a Zola Markdown file."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "+++":
        raise AssertionError(f"{path.relative_to(REPO_ROOT)} has no TOML front matter")

    try:
        closing_line = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "+++"
        )
    except StopIteration as error:
        raise AssertionError(
            f"{path.relative_to(REPO_ROOT)} has unterminated TOML front matter"
        ) from error

    try:
        return tomllib.loads("\n".join(lines[1:closing_line]))
    except tomllib.TOMLDecodeError as error:
        raise AssertionError(
            f"Invalid TOML in {path.relative_to(REPO_ROOT)}: {error}"
        ) from error


def content_path(path: Path) -> str:
    """Return the slash-separated path expected by Zola's get_section()."""
    return path.relative_to(CONTENT_ROOT).as_posix()


class PostSeriesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.blog_files = sorted(BLOG_ROOT.rglob("*.md"))
        cls.front_matter = {path: load_front_matter(path) for path in cls.blog_files}
        cls.series_sections = {
            path: metadata
            for path, metadata in cls.front_matter.items()
            if path.name == "_index.md"
            and (
                metadata.get("template") == "series.html"
                or metadata.get("extra", {}).get("series") is True
            )
        }

    def test_repository_contains_a_series(self) -> None:
        self.assertTrue(
            self.series_sections,
            "No post-series section was found under content/blog",
        )

    def test_series_sections_follow_the_section_convention(self) -> None:
        for section, metadata in self.series_sections.items():
            label = content_path(section)
            with self.subTest(series=label):
                self.assertEqual(metadata.get("template"), "series.html")
                self.assertEqual(metadata.get("page_template"), "blog-page.html")
                self.assertEqual(metadata.get("sort_by"), "weight")
                self.assertIs(metadata.get("transparent"), True)
                self.assertIs(metadata.get("extra", {}).get("series"), True)
                self.assertIsInstance(metadata.get("title"), str)
                self.assertTrue(metadata.get("title", "").strip())
                self.assertIsInstance(metadata.get("description"), str)
                self.assertTrue(metadata.get("description", "").strip())

    def test_series_members_point_back_to_their_section(self) -> None:
        for section in self.series_sections:
            expected_pointer = content_path(section)
            members = sorted(
                path
                for path in section.parent.rglob("*.md")
                if path.name != "_index.md"
            )
            with self.subTest(series=expected_pointer):
                self.assertTrue(members, f"{expected_pointer} has no member posts")

            for member in members:
                metadata = self.front_matter[member]
                with self.subTest(member=content_path(member)):
                    self.assertEqual(
                        metadata.get("extra", {}).get("series"),
                        expected_pointer,
                        "Series members must explicitly reference their section",
                    )

    def test_series_pointers_resolve_to_the_containing_series(self) -> None:
        for page, metadata in self.front_matter.items():
            pointer = metadata.get("extra", {}).get("series")
            if not isinstance(pointer, str):
                continue

            target = CONTENT_ROOT / Path(pointer)
            with self.subTest(member=content_path(page), series=pointer):
                self.assertIn(
                    target,
                    self.series_sections,
                    "extra.series must reference an existing series _index.md",
                )
                self.assertTrue(
                    page.is_relative_to(target.parent),
                    "A series member must live below the referenced series directory",
                )

    def test_series_members_have_unique_weights_and_output_paths(self) -> None:
        all_output_paths: dict[str, Path] = {}

        for section in self.series_sections:
            weights: dict[int, Path] = {}
            members = sorted(
                path
                for path in section.parent.rglob("*.md")
                if path.name != "_index.md"
            )

            for member in members:
                metadata = self.front_matter[member]
                label = content_path(member)
                weight = metadata.get("weight")
                output_path = metadata.get("path")

                with self.subTest(member=label):
                    self.assertIsInstance(weight, int, "Series members need an integer weight")
                    self.assertGreater(weight, 0, "Series member weights must be positive")
                    self.assertNotIn(
                        weight,
                        weights,
                        f"Duplicate weight also used by {content_path(weights[weight])}"
                        if weight in weights
                        else "Duplicate series weight",
                    )
                    if isinstance(weight, int):
                        weights[weight] = member

                    self.assertIsInstance(
                        output_path,
                        str,
                        "Series members need an explicit stable output path",
                    )
                    self.assertRegex(
                        output_path or "",
                        r"^blog/[^/].*[^/]$",
                        "Series output paths must use the form blog/post-slug",
                    )
                    self.assertNotIn(
                        output_path,
                        all_output_paths,
                        f"Duplicate output path also used by "
                        f"{content_path(all_output_paths[output_path])}"
                        if output_path in all_output_paths
                        else "Duplicate series output path",
                    )
                    if isinstance(output_path, str):
                        all_output_paths[output_path] = member


if __name__ == "__main__":
    unittest.main()
