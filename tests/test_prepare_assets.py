"""Asset ordering and edit coalescing, without expensive font jobs or sleeps."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import prepare_assets as assets


REPO_ROOT = Path(__file__).resolve().parents[1]


class PrepareAssetsTests(unittest.TestCase):
    def test_search_is_ready_before_fonts_collect_characters(self) -> None:
        order = []
        with (
            mock.patch.object(assets.search, "ensure_index", side_effect=lambda: order.append("search")),
            mock.patch.object(assets.fonts, "source_characters", side_effect=lambda: order.append("characters") or {65}),
            mock.patch.object(assets.fonts, "ensure_subsets", return_value=([], True)) as fonts,
            mock.patch.object(assets.fonts, "print_summary"),
        ):
            assets.prepare_assets()
        self.assertEqual(order, ["search", "characters"])
        fonts.assert_called_once_with({65})

    def test_cache_key_prepares_search_without_building_fonts(self) -> None:
        with (
            mock.patch("sys.argv", ["prepare_assets", "--print-font-cache-key"]),
            mock.patch.object(assets.search, "ensure_index") as search,
            mock.patch.object(assets.fonts, "source_characters", return_value={65}),
            mock.patch.object(assets.fonts, "cache_key", return_value="font-cache-key"),
            mock.patch.object(assets.fonts, "ensure_subsets") as fonts,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(assets.main(), 0)
        search.assert_called_once_with()
        fonts.assert_not_called()
        self.assertEqual(output.getvalue(), "font-cache-key\n")

    def test_fingerprint_tracks_source_edits_and_deletions_but_ignores_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "post.md"
            generated = Path(directory) / "index.json"
            source.write_bytes(b"abc")
            generated.write_bytes(b"old index")
            with (
                mock.patch.object(assets.search, "OUTPUT_PATH", generated),
                mock.patch.object(assets.fonts, "source_text_files", return_value=[source, generated]),
            ):
                before = assets.source_fingerprint()
                generated.write_bytes(b"new index")
                self.assertEqual(before, assets.source_fingerprint())
                # Reordering the same characters must still update search.
                source.write_bytes(b"cba")
                self.assertNotEqual(before, assets.source_fingerprint())
                source.unlink()
                self.assertEqual(assets.source_fingerprint(), ())

    def test_watcher_coalesces_rapid_edits_and_skips_unchanged_sources(self) -> None:
        first, second, third = [(("page.md", value),) for value in (b"a", b"b", b"c")]
        with (
            mock.patch.object(assets, "source_fingerprint", side_effect=[first, second, second, second, third, third]),
            mock.patch.object(assets, "prepare_assets") as prepare,
            mock.patch.object(assets.time, "sleep", side_effect=[None] * 5 + [KeyboardInterrupt]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(assets.watch(0.75), 0)
        self.assertEqual(prepare.call_count, 2)

    def test_watcher_retries_invalid_content_and_recovers(self) -> None:
        with (
            mock.patch.object(assets, "source_fingerprint", return_value=(("page.md", b"a"),)),
            mock.patch.object(assets, "prepare_assets", side_effect=[ValueError("partial save"), ValueError("partial save"), None]) as prepare,
            mock.patch.object(assets.time, "sleep", side_effect=[None] * 4 + [KeyboardInterrupt]),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            self.assertEqual(assets.watch(0.75), 0)
        self.assertEqual(prepare.call_count, 3)
        self.assertEqual(errors.getvalue().count("partial save"), 1)

    def test_development_build_and_ci_use_shared_preparation(self) -> None:
        unix_wrapper = (REPO_ROOT / "serve.sh").read_text(encoding="utf-8")
        windows_wrapper = (REPO_ROOT / "serve.ps1").read_text(encoding="utf-8")
        unix_build = (REPO_ROOT / "build.sh").read_text(encoding="utf-8")
        windows_build = (REPO_ROOT / "build.ps1").read_text(encoding="utf-8")
        workflow = (REPO_ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")

        self.assertIn("scripts.prepare_assets --watch", unix_wrapper)
        self.assertIn("scripts.prepare_assets', '--watch", windows_wrapper)
        for entry_point in (unix_wrapper, windows_wrapper, unix_build, windows_build):
            self.assertIn("python -m scripts.prepare_assets", entry_point)
        self.assertIn("build_font_subsets.py --check-rendered public", unix_build)
        self.assertIn("build_font_subsets.py --check-rendered public", windows_build)
        self.assertIn("run: ./build.sh", workflow)
        self.assertIn("scripts.prepare_assets --print-font-cache-key", workflow)
        self.assertNotIn("git diff --exit-code -- static/search_index.zh.json", workflow)

    def test_generated_artifacts_are_ignored(self) -> None:
        ignore_rules = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        expected = {
            ".font-subsets-manifest.json",
            "static/fonts/generated/",
            assets.search.INDEX_PATH.as_posix(),
        }
        self.assertTrue(expected.issubset(ignore_rules))


if __name__ == "__main__":
    unittest.main()
