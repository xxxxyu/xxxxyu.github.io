"""Checks for the automatically generated web-font subsets."""

from __future__ import annotations

import hashlib
import fnmatch
import json
import os
import re
import tempfile
import tomllib
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_font_subsets


REPO_ROOT = Path(__file__).resolve().parents[1]


class FontSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/build_font_subsets.py")],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stdout + result.stderr)

    def test_unchanged_subsets_use_the_incremental_manifest(self) -> None:
        manifest_path = REPO_ROOT / ".font-subsets-manifest.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertIn("inputs", manifest)
        self.assertEqual(len(manifest.get("outputs", {})), len(build_font_subsets.FONT_TARGETS) + 2)

        outputs = [REPO_ROOT / path for path in manifest["outputs"]]
        mtimes = {path: path.stat().st_mtime_ns for path in outputs}
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/build_font_subsets.py")],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already current", result.stdout)
        self.assertEqual(mtimes, {path: path.stat().st_mtime_ns for path in outputs})

    def test_generated_subsets_cover_all_site_sources(self) -> None:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/build_font_subsets.py"), "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        characters = build_font_subsets.source_characters()
        for target in build_font_subsets.FONT_TARGETS:
            with self.subTest(font=target.output.name):
                self.assertTrue(target.output.is_file())
                self.assertLess(target.output.stat().st_size, target.source.stat().st_size)
                required = build_font_subsets.target_characters(target, characters)
                supported = build_font_subsets.font_codepoints(target.source)
                self.assertFalse((required & supported) - build_font_subsets.font_codepoints(target.output))

    def test_process_liveness_check_handles_current_and_missing_pids(self) -> None:
        self.assertTrue(build_font_subsets.process_is_running(os.getpid()))
        self.assertFalse(build_font_subsets.process_is_running(0))

    def test_generated_font_urls_match_bytes_and_preloads_are_selective(self) -> None:
        css = build_font_subsets.FONT_CSS.read_text()
        catalog = json.loads(build_font_subsets.FONT_CATALOG.read_text())
        for entry in catalog["fonts"]:
            url, digest = entry["url"].split("?v=")
            path = REPO_ROOT / "static" / url.lstrip("/")
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest()[:12])
            self.assertIn(entry["url"], css)
        self.assertEqual(len(re.findall(r"unicode-range:", css)), len(build_font_subsets.FONT_TARGETS))
        base = (REPO_ROOT / "templates/base.html").read_text()
        self.assertIn('font.style == "normal"', base)
        self.assertIn('not font.cjk or lang == "zh"', base)
        self.assertNotIn('SC Full', (REPO_ROOT / "sass/_variables.scss").read_text())

    def test_coverage_partitions_one_normalized_snapshot(self) -> None:
        points = build_font_subsets.characters_from_texts(["e\u0301 &rarr; 中"])
        self.assertLessEqual(set(range(0x20, 0x7F)), points)
        self.assertIn(ord("é"), points)
        self.assertIn(ord("→"), points)
        self.assertIn(ord("中"), points)
        for target in build_font_subsets.FONT_TARGETS:
            # Partitioning must not read mutable source files a second time.
            with mock.patch.object(build_font_subsets, "source_text_files", side_effect=AssertionError):
                selected = build_font_subsets.target_characters(target, points)
            self.assertEqual(ord("中") in selected, target.cjk)
            self.assertEqual(ord("é") in selected, not target.cjk)

    def test_failed_atomic_replace_keeps_previous_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "font.woff2"
            path.write_bytes(b"previous")
            with mock.patch.object(Path, "replace", side_effect=OSError("write failed")):
                with self.assertRaises(OSError):
                    build_font_subsets.atomic_write(path, b"new")
            self.assertEqual(path.read_bytes(), b"previous")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_deployment_keeps_generated_fonts_and_excludes_source_fonts(self) -> None:
        config = tomllib.loads((REPO_ROOT / "config.toml").read_text())
        patterns = config["ignored_static"]
        for target in build_font_subsets.FONT_TARGETS:
            for path, excluded in ((target.source, True), (target.output, False)):
                relative = path.relative_to(REPO_ROOT / "static").as_posix()
                with self.subTest(path=relative):
                    self.assertEqual(any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns), excluded)


if __name__ == "__main__":
    unittest.main()
