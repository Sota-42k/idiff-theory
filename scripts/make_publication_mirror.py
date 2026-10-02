"""Assemble the publication mirror: the implementation and the production results, nothing else.

  uv run python scripts/make_publication_mirror.py --out ../recovery-publication
  uv run python scripts/make_publication_mirror.py --check        # verify without writing

Development artefacts stay in the working repository. The point of this script is not the
copying, which is trivial, but the check afterwards: a mirror that still mentions a file it
does not contain is a mirror that cannot be read on its own, and that is easy to produce by
hand and hard to notice. `--check` fails on any such reference, and on any tracked file that
neither list claims, so a new file cannot be forgotten into limbo.
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Everything the mirror carries. Implementation, the configurations that were run, the
# documents that explain them, and the licence trail.
PUBLISHED = (
    "README.md", "LICENSE", "NOTICE", "pyproject.toml", "uv.lock", ".gitignore", ".python-version",
    "src/recovery_core/", "src/recovery_estimators/", "src/recovery_reporting/", "tests/",
    "scripts/run_recovery.py", "scripts/analyse_recovery.py",
    "scripts/factorization_gap.py", "scripts/select_potential.py",
    "scripts/run_estimators.py", "scripts/analyse_estimators.py", "scripts/run_classifier_checkpoints.py",
    "scripts/reproduce.py", "scripts/prepare_publication_data.py", "scripts/make_publication_mirror.py",
    "configs/recovery-n2.yaml", "configs/recovery-n3.yaml", "configs/estimators.yaml",
    "docs/experiment-design.md", "docs/provenance.md", "docs/equivalence_check.py",
    "docs/results.md", "docs/estimators.md", "docs/figures.md", "docs/reproducibility.md",
    "data/", "figures/",
)

WITHHELD = (
    "CLAUDE.md", "AGENTS.md", "configs/recovery-v1-superseded.yaml",
    "docs/pilot.md", "docs/pilot.json", "docs/pilot-selection.json",
    "scripts/pilot_recovery.py", "scripts/extra_sensitivity.py",
)


def tracked() -> list[str]:
    if (ROOT / ".git").exists():
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return sorted(set(out.rstrip("\0").split("\0")))
    # A standalone export has no .git directory. Ignore only transient run/cache files.
    ignored = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", "results", "build", "dist"}
    return sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()
                  and not set(p.relative_to(ROOT).parts) & ignored and p.name != ".DS_Store")


def classify(paths: list[str]) -> tuple[list[str], list[str]]:
    published, unclaimed = [], []
    for path in paths:
        if any(path == p or (p.endswith("/") and path.startswith(p)) for p in PUBLISHED):
            published.append(path)
        elif path in WITHHELD:
            continue
        else:
            unclaimed.append(path)
    return published, unclaimed


def dangling(published: list[str]) -> dict[str, list[str]]:
    """References from mirrored text to files the mirror does not carry."""
    stems = {Path(p).name for p in WITHHELD} | {p for p in WITHHELD}
    found: dict[str, list[str]] = {}
    for path in published:
        if path == "scripts/make_publication_mirror.py":
            continue  # This script declares the intentionally withheld filenames.
        full = ROOT/path
        if full.suffix not in (".md", ".py", ".yaml", ".toml", ".cfg", ".txt"):
            continue
        text = full.read_text(errors="ignore")
        hits = sorted({s for s in stems
                       if re.search(r"(?<![\w/.-])" + re.escape(s) + r"(?![\w.-])", text)})
        if hits:
            found[path] = hits
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true", help="regenerate and verify; write nothing")
    args = parser.parse_args()
    if args.check == (args.out is not None):
        parser.error("give exactly one of --out or --check")
    published, unclaimed = classify(tracked())
    problems = []
    if unclaimed:
        problems.append("files claimed by neither list: " + ", ".join(unclaimed))
    for path, hits in dangling(published).items():
        problems.append(f"{path} refers to withheld " + ", ".join(hits))
    missing = [p for p in PUBLISHED if not (ROOT / p).exists()]
    if missing:
        problems.append("listed for publication but absent: " + ", ".join(missing))
    if problems:
        raise SystemExit("\n".join(problems))
    from reproduce import verify
    verify()
    if args.check:
        print(f"Publication ready: {len(published)} files; selected dataset and generated outputs verified")
        return
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new output directory")
    for name in published:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    print(f"Wrote {len(published)} publication files to {out}")


if __name__ == "__main__":
    main()
