"""Repository checks for CJK emphasis pairs that CommonMark renders literally."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOT = REPO_ROOT / "content"

# Fullwidth and CJK punctuation that commonly ends an emphasized span.
CJK_PUNCT = "。，、；：？！）》」』】〕〉”’·…—～%％"
# A closing delimiter stays closable when the next character is whitespace or
# punctuation; it stops being closable when followed directly by a letter or
# digit (CJK ideographs included).
SAFE_AFTER_CLOSER = r"[^\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]"


def strip_code_spans(line: str) -> str:
    """Blank inline code spans so their delimiters are never inspected."""
    return re.sub(r"`+[^`]*`+", lambda match: " " * len(match.group()), line)


def closing_delimiter_violations(text: str) -> list[tuple[int, str, str]]:
    """Return (line_number, delimiter, context) for delimiters that cannot close.

    Zola renders Markdown with comrak, which follows the CommonMark emphasis
    rules. A ``**`` or ``*`` delimiter in closing position that is preceded by
    CJK punctuation and directly followed by an ordinary character is
    left-flanking and cannot close, so the pair stays literal in the published
    HTML, as in ``记下：**结论。**按 replay``. Pairing walks each line's
    delimiters in order, assuming well-formed input.
    """
    violations: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            continue
        line = strip_code_spans(line)
        for delimiter in ("**", "*"):
            positions = [match.start() for match in re.finditer(re.escape(delimiter), line)]
            for position in positions[1::2]:
                before = line[position - 1] if position > 0 else ""
                after_index = position + len(delimiter)
                after = line[after_index] if after_index < len(line) else ""
                if not before or not after:
                    continue
                if before in CJK_PUNCT and not re.match(SAFE_AFTER_CLOSER, after):
                    context = line[max(0, position - 14) : position + 16]
                    violations.append((line_number, delimiter, context))
    return violations


class CjkEmphasisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.markdown_files = sorted(CONTENT_ROOT.rglob("*.md"))
        cls.violations = {
            path: closing_delimiter_violations(path.read_text(encoding="utf-8"))
            for path in cls.markdown_files
        }

    def test_repository_contains_markdown_content(self) -> None:
        self.assertTrue(self.markdown_files, "expected Markdown content files")

    def test_no_unclosable_cjk_emphasis_in_content(self) -> None:
        for path, violations in self.violations.items():
            with self.subTest(source=path.relative_to(REPO_ROOT)):
                self.assertEqual(
                    violations,
                    [],
                    "Closing emphasis delimiters preceded by CJK punctuation and "
                    "followed directly by text render literally under CommonMark. "
                    "Move the punctuation outside the emphasis or add a space after "
                    "the closing delimiter.",
                )

    def test_detector_catches_known_failure_and_passes_good_input(self) -> None:
        bad = "记下：**开环动作误差预测不了闭环成功率。**按 replay 后续文字。"
        self.assertEqual(len(closing_delimiter_violations(bad)), 1)

        good_cases = (
            "记下：**开环动作误差预测不了闭环成功率**。按 replay 后续文字。",
            "列表项 **加粗文字。** 后续文字。",
            "获奖（2024），**Vec-LUT** 是一项工作。",
            "**查找表**（lookup table, LUT）内核在单 token 生成时表现很好。",
            "代码内的 **标记** `不**参与**配对` 不受影响。",
        )
        for case in good_cases:
            with self.subTest(case=case):
                self.assertEqual(closing_delimiter_violations(case), [])


if __name__ == "__main__":
    unittest.main()
