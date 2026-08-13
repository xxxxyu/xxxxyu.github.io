"""Checks for the automatically generated CJK web-font subsets."""

from __future__ import annotations

import hashlib
import json
import os
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
        self.assertEqual(len(manifest.get("outputs", {})), 2)

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

        pairs = (
            (
                REPO_ROOT / "static/fonts/noto-serif-sc/NotoSerifSC-SiteSubset.woff2",
                REPO_ROOT / "static/fonts/noto-serif-sc/NotoSerifSC-VariableFont_wght.woff2",
            ),
            (
                REPO_ROOT / "static/fonts/noto-sans-sc/NotoSansSC-SiteSubset.woff2",
                REPO_ROOT / "static/fonts/noto-sans-sc/NotoSansSC-VariableFont_wght.woff2",
            ),
        )
        for generated, source in pairs:
            with self.subTest(font=generated.name):
                self.assertTrue(generated.is_file())
                self.assertLess(generated.stat().st_size, source.stat().st_size)

    def test_watcher_snapshot_tracks_source_content(self) -> None:
        source = REPO_ROOT / "content/about/_index.md"
        first = build_font_subsets.source_snapshot()
        with mock.patch.object(
            build_font_subsets,
            "source_text_files",
            return_value=[source],
        ):
            isolated = build_font_subsets.source_snapshot()

        self.assertIn(str(source.relative_to(REPO_ROOT)), {entry[0] for entry in first})
        self.assertEqual(isolated[0][0], str(source.relative_to(REPO_ROOT)))
        self.assertEqual(
            isolated[0][1],
            hashlib.sha256(source.read_text(encoding="utf-8").encode("utf-8")).digest(),
        )
        self.assertEqual(isolated[0][2], source.read_text(encoding="utf-8"))

    def test_process_liveness_check_handles_current_and_missing_pids(self) -> None:
        self.assertTrue(build_font_subsets.process_is_running(os.getpid()))
        self.assertFalse(build_font_subsets.process_is_running(0))

    def test_subset_loading_keeps_full_fonts_as_defensive_fallbacks(self) -> None:
        variables = (REPO_ROOT / "sass/_variables.scss").read_text(encoding="utf-8")
        base = (REPO_ROOT / "templates/base.html").read_text(encoding="utf-8")

        for family in ("Serif", "Sans"):
            subset_name = f'Noto {family} SC Site'
            fallback_name = f'Noto {family} SC Full'
            self.assertIn(subset_name, variables)
            self.assertIn(fallback_name, variables)
            self.assertLess(variables.index(subset_name), variables.index(fallback_name))
            self.assertIn(f"Noto{family}SC-SiteSubset.woff2", base)
            self.assertNotIn(f"Noto{family}SC-VariableFont_wght.woff2", base)

        self.assertIn(
            'rel="preload" href="/fonts/source-serif-4/SourceSerif4Variable-Italic.woff2"',
            base,
        )

    def test_development_and_build_entry_points_manage_subsets(self) -> None:
        unix_wrapper = (REPO_ROOT / "serve.sh").read_text(encoding="utf-8")
        windows_wrapper = (REPO_ROOT / "serve.ps1").read_text(encoding="utf-8")
        unix_build = (REPO_ROOT / "build.sh").read_text(encoding="utf-8")
        windows_build = (REPO_ROOT / "build.ps1").read_text(encoding="utf-8")
        workflow = (REPO_ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")

        self.assertIn("build_font_subsets.py --watch", unix_wrapper)
        self.assertIn("build_font_subsets.py', '--watch", windows_wrapper)
        self.assertIn("build_font_subsets.py --check-rendered public", unix_build)
        self.assertIn("build_font_subsets.py --check-rendered public", windows_build)
        self.assertIn("run: ./build.sh", workflow)

    def test_generated_artifacts_are_ignored(self) -> None:
        ignore_rules = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        expected = {
            ".font-subsets-manifest.json",
            "static/fonts/noto-sans-sc/NotoSansSC-SiteSubset.woff2",
            "static/fonts/noto-serif-sc/NotoSerifSC-SiteSubset.woff2",
        }
        self.assertTrue(expected.issubset(ignore_rules))


if __name__ == "__main__":
    unittest.main()
