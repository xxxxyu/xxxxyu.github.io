"""Build self-hosted Noto CJK subsets from the site's textual sources.

The source fonts remain available as a defensive fallback. Normal pages use the
small generated subsets; development and deployment entry points regenerate
them automatically whenever the set of CJK characters changes.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import signal
import sys
import tempfile
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

from fontTools import subset
from fontTools import __version__ as fonttools_version
from fontTools.ttLib import TTFont


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_ROOTS = (
    REPO_ROOT / "content",
    REPO_ROOT / "data",
    REPO_ROOT / "templates",
    REPO_ROOT / "sass",
    REPO_ROOT / "static" / "js",
)
TEXT_FILES = (
    REPO_ROOT / "config.toml",
    REPO_ROOT / "static" / "search_index.zh.json",
)
TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".scss", ".toml", ".txt", ".xml"}
RENDERED_SUFFIXES = {".html", ".json", ".txt", ".xml"}
MANIFEST_PATH = REPO_ROOT / ".font-subsets-manifest.json"
LOCK_PATH = REPO_ROOT / ".font-subsets.lock"
SUBSET_FORMAT_VERSION = 1


@dataclass(frozen=True)
class FontTarget:
    label: str
    source: Path
    output: Path


FONT_TARGETS = (
    FontTarget(
        "Noto Serif SC",
        REPO_ROOT / "static/fonts/noto-serif-sc/NotoSerifSC-VariableFont_wght.woff2",
        REPO_ROOT / "static/fonts/noto-serif-sc/NotoSerifSC-SiteSubset.woff2",
    ),
    FontTarget(
        "Noto Sans SC",
        REPO_ROOT / "static/fonts/noto-sans-sc/NotoSansSC-VariableFont_wght.woff2",
        REPO_ROOT / "static/fonts/noto-sans-sc/NotoSansSC-SiteSubset.woff2",
    ),
)


# Scripts and punctuation covered by the site's Chinese fallback fonts. Keeping
# this independent of a fixed list of Han characters lets new posts expand the
# generated subset automatically.
CJK_RANGES = (
    (0x2E80, 0x33FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFE10, 0xFE1F),
    (0xFE30, 0xFE4F),
    (0xFF00, 0xFFEF),
    (0x20000, 0x2FA1F),
    (0x30000, 0x323AF),
)


def is_cjk_character(character: str) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in CJK_RANGES)


def textual_files(root: Path, suffixes: set[str]) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in suffixes else []
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def characters_from_texts(sources: list[str]) -> set[int]:
    characters: set[int] = set()
    for source in sources:
        normalized = unicodedata.normalize("NFC", html.unescape(source))
        characters.update(ord(character) for character in normalized if is_cjk_character(character))
    return characters


def characters_in(paths: list[Path]) -> set[int]:
    sources: list[str] = []
    for path in paths:
        try:
            sources.append(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            # A live-editing rename may happen between discovery and reading.
            continue
    return characters_from_texts(sources)


def source_characters() -> set[int]:
    return characters_in(source_text_files())


def source_text_files() -> list[Path]:
    paths: list[Path] = []
    for root in TEXT_ROOTS:
        paths.extend(textual_files(root, TEXT_SUFFIXES))
    paths.extend(path for path in TEXT_FILES if path.is_file())
    return sorted(set(paths))


def source_snapshot() -> tuple[tuple[str, bytes, str], ...]:
    """Read source bytes once for change detection and character extraction."""
    snapshot: list[tuple[str, bytes, str]] = []
    for path in source_text_files():
        try:
            source = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # A live-editing rename may happen between discovery and reading.
            continue
        digest = hashlib.sha256(source.encode("utf-8")).digest()
        snapshot.append((str(path.relative_to(REPO_ROOT)), digest, source))
    return tuple(snapshot)


def rendered_characters(root: Path) -> set[int]:
    return characters_in(textual_files(root, RENDERED_SUFFIXES))


def font_codepoints(path: Path) -> set[int]:
    with TTFont(path, lazy=True) as font:
        return set(font.getBestCmap() or {})


def describe_codepoints(codepoints: set[int], limit: int = 12) -> str:
    entries = [f"{chr(value)} (U+{value:04X})" for value in sorted(codepoints)[:limit]]
    if len(codepoints) > limit:
        entries.append(f"… and {len(codepoints) - limit} more")
    return ", ".join(entries)


def validate_source_fonts() -> None:
    for target in FONT_TARGETS:
        if not target.source.is_file():
            raise RuntimeError(f"Missing source font: {target.source.relative_to(REPO_ROOT)}")


def process_is_running(pid: int) -> bool:
    """Check a lock owner's liveness without signaling it on Windows."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            # Access denied still proves that a process owns the PID.
            return ctypes.get_last_error() == 5
        exit_code = ctypes.c_ulong()
        try:
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def generation_lock() -> object:
    """Serialize writers without adding a platform-specific dependency."""
    announced_wait = False
    while True:
        try:
            LOCK_PATH.mkdir()
            break
        except FileExistsError:
            try:
                owner = int((LOCK_PATH / "pid").read_text(encoding="ascii"))
            except FileNotFoundError:
                # mkdir and writing the owner PID are separate operations. Give
                # a newly-created lock time to finish initialization.
                if time.time() - LOCK_PATH.stat().st_mtime < 2:
                    time.sleep(0.1)
                    continue
                owner = 0
            except ValueError:
                owner = 0
            if owner and not process_is_running(owner):
                owner = 0
            if owner == 0:
                (LOCK_PATH / "pid").unlink(missing_ok=True)
                try:
                    LOCK_PATH.rmdir()
                except OSError:
                    time.sleep(0.1)
                continue
            if not announced_wait:
                print("Another font subset build is running; waiting…", flush=True)
                announced_wait = True
            time.sleep(0.25)
    try:
        (LOCK_PATH / "pid").write_text(str(os.getpid()), encoding="ascii")
        yield
    finally:
        (LOCK_PATH / "pid").unlink(missing_ok=True)
        try:
            LOCK_PATH.rmdir()
        except FileNotFoundError:
            pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(required: set[int]) -> dict[str, object]:
    validate_source_fonts()
    character_digest = hashlib.sha256(
        ",".join(f"{codepoint:X}" for codepoint in sorted(required)).encode("ascii")
    ).hexdigest()
    return {
        "format_version": SUBSET_FORMAT_VERSION,
        "generator_sha256": sha256(Path(__file__)),
        "fonttools_version": fonttools_version,
        "characters_sha256": character_digest,
        "source_fonts": {
            str(target.source.relative_to(REPO_ROOT)): sha256(target.source)
            for target in FONT_TARGETS
        },
    }


def cache_key(required: set[int]) -> str:
    serialized = json.dumps(input_fingerprint(required), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_manifest() -> dict[str, object] | None:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return manifest if isinstance(manifest, dict) else None


def subsets_are_current(required: set[int]) -> bool:
    manifest = load_manifest()
    if manifest is None or manifest.get("inputs") != input_fingerprint(required):
        return False
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        return False
    for target in FONT_TARGETS:
        relative = str(target.output.relative_to(REPO_ROOT))
        if not target.output.is_file() or outputs.get(relative) != sha256(target.output):
            return False
    return True


def write_manifest(required: set[int]) -> None:
    manifest = {
        "inputs": input_fingerprint(required),
        "outputs": {
            str(target.output.relative_to(REPO_ROOT)): sha256(target.output)
            for target in FONT_TARGETS
        },
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(suffix=".json", dir=MANIFEST_PATH.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(rendered, encoding="utf-8", newline="\n")
        temporary.chmod(0o644)
        temporary.replace(MANIFEST_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def subset_bytes(target: FontTarget, codepoints: set[int]) -> bytes:
    options = subset.Options()
    options.flavor = "woff2"
    options.layout_features = ["*"]
    options.name_IDs = ["*"]
    options.name_languages = ["*"]
    options.notdef_outline = True
    options.recalc_timestamp = False
    options.recommended_glyphs = True

    font = subset.load_font(str(target.source), options)
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(unicodes=codepoints)
    subsetter.subset(font)

    target.output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(suffix=".woff2", dir=target.output.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        subset.save_font(font, str(temporary), options)
        return temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)


def subset_job(job: tuple[FontTarget, set[int]]) -> tuple[FontTarget, bytes]:
    target, codepoints = job
    return target, subset_bytes(target, codepoints)


def ignore_interrupt_in_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def build_subsets(required: set[int]) -> list[Path]:
    if not required:
        raise RuntimeError("No CJK characters were found in the site sources")
    validate_source_fonts()
    jobs: list[tuple[FontTarget, set[int]]] = []
    for target in FONT_TARGETS:
        supported = required & font_codepoints(target.source)
        if not supported:
            raise RuntimeError(f"{target.label} supports none of the CJK characters used by the site")
        jobs.append((target, supported))

    changed: list[Path] = []
    pool = Pool(processes=len(jobs), initializer=ignore_interrupt_in_worker)
    try:
        generated_fonts = pool.map(subset_job, jobs)
    except KeyboardInterrupt:
        # The parent owns interruption. Stop workers promptly instead of
        # waiting for both large font jobs to finish during shell startup.
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()
    for target, generated in generated_fonts:
        current = target.output.read_bytes() if target.output.is_file() else None
        if current != generated:
            target.output.write_bytes(generated)
            changed.append(target.output)
    return changed


def ensure_subsets(required: set[int]) -> tuple[list[Path], bool]:
    if subsets_are_current(required):
        return [], True
    with generation_lock():
        # A concurrent process may have completed while this one waited.
        if subsets_are_current(required):
            return [], True
        print("CJK character set changed; rebuilding font subsets…", flush=True)
        changed = build_subsets(required)
        write_manifest(required)
        return changed, False


def validate_subsets(required: set[int], *, context: str) -> None:
    for target in FONT_TARGETS:
        if not target.output.is_file():
            raise RuntimeError(
                f"Missing generated font: {target.output.relative_to(REPO_ROOT)}; run this script"
            )
        expected = required & font_codepoints(target.source)
        missing = expected - font_codepoints(target.output)
        if missing:
            raise RuntimeError(
                f"{target.output.relative_to(REPO_ROOT)} does not cover {context}: "
                f"{describe_codepoints(missing)}"
            )


def print_summary(required: set[int], changed: list[Path], *, cached: bool = False) -> None:
    state = "already current" if cached else ("updated" if changed else "regenerated")
    sizes = ", ".join(
        f"{target.output.name} {target.output.stat().st_size / 1024:.0f} KiB"
        for target in FONT_TARGETS
    )
    print(f"CJK font subsets {state}: {len(required)} characters; {sizes}", flush=True)


def watch(interval: float) -> int:
    previous_fingerprint: tuple[tuple[str, bytes], ...] | None = None
    previous: frozenset[int] | None = None
    last_error: str | None = None
    print("Watching site sources for new CJK characters…", flush=True)
    try:
        while True:
            snapshot = source_snapshot()
            fingerprint = tuple((path, digest) for path, digest, _ in snapshot)
            if fingerprint == previous_fingerprint:
                time.sleep(interval)
                continue
            try:
                required = frozenset(characters_from_texts([source for _, _, source in snapshot]))
                if required != previous:
                    changed, cached = ensure_subsets(set(required))
                    print_summary(set(required), changed, cached=cached)
                previous = required
                previous_fingerprint = fingerprint
                last_error = None
            except Exception as error:  # keep the development server alive and retry
                message = str(error)
                if message != last_error:
                    print(f"Font subset update failed: {message}", file=sys.stderr, flush=True)
                    last_error = message
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify that generated subsets cover source text")
    parser.add_argument(
        "--check-rendered",
        type=Path,
        metavar="DIRECTORY",
        help="verify that generated subsets cover the rendered site",
    )
    parser.add_argument("--watch", action="store_true", help="regenerate when the source character set changes")
    parser.add_argument("--force", action="store_true", help="rebuild even when the incremental manifest is current")
    parser.add_argument(
        "--print-cache-key",
        action="store_true",
        help="print a stable cache key for the current subset inputs",
    )
    parser.add_argument("--interval", type=float, default=0.75, help=argparse.SUPPRESS)
    args = parser.parse_args()

    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    try:
        if args.watch:
            return watch(args.interval)

        required = source_characters()
        if args.print_cache_key:
            print(cache_key(required))
            return 0

        if args.check_rendered is not None:
            rendered_root = args.check_rendered.resolve()
            if not rendered_root.is_dir():
                raise RuntimeError(f"Rendered site directory does not exist: {rendered_root}")
            required |= rendered_characters(rendered_root)

        if args.check:
            if not subsets_are_current(required):
                raise RuntimeError("CJK font subsets are stale; run this script without --check")
            validate_subsets(required, context="site sources")
            return 0

        if args.check_rendered is not None:
            validate_subsets(required, context="the rendered site" if args.check_rendered else "site sources")
            return 0

        if args.force:
            with generation_lock():
                print("Forcing CJK font subset rebuild…", flush=True)
                changed = build_subsets(required)
                write_manifest(required)
            cached = False
        else:
            changed, cached = ensure_subsets(required)
        print_summary(required, changed, cached=cached)
        return 0
    except KeyboardInterrupt:
        print("Font subset generation cancelled.", file=sys.stderr)
        return 130
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
