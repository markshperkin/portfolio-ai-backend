"""Walk a knowledge corpus directory, parse frontmatter, return structured docs."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

REQUIRED_FIELDS = {"title", "category", "tags", "last_updated"}
VALID_CATEGORIES = {"project", "experience", "education", "skills", "paper", "meta"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SKIP_FILES = {"README.md", "INDEX.md"}
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
H1_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)

# Maps top-level corpus directory names to document categories
_DIR_TO_CATEGORY: dict[str, str] = {
    "projects": "project",
    "research": "paper",
    "education": "education",
    "experience": "experience",
    "skills": "skills",
}


@dataclass
class KnowledgeDoc:
    path: str          # relative to corpus root, e.g. "projects/tutor-ai.md"
    title: str
    category: str
    tags: list[str]
    last_updated: str
    weight: float
    body: str          # everything after the frontmatter block
    metadata: dict[str, Any]


def load_corpus(corpus_root: Path) -> list[KnowledgeDoc]:
    docs: list[KnowledgeDoc] = []
    for md_file in sorted(corpus_root.rglob("*.md")):
        rel = md_file.relative_to(corpus_root)
        # Skip files in .github/ and scripts/ and top-level skips
        parts = rel.parts
        if parts[0] in {".github", "scripts"} or md_file.name in SKIP_FILES:
            continue
        doc = _parse_file(md_file, str(rel))
        docs.append(doc)
    return docs


def _parse_file(path: Path, rel_path: str) -> KnowledgeDoc:
    raw = path.read_text(encoding="utf-8")
    rel = Path(rel_path)

    m = FRONTMATTER_RE.match(raw)
    if m:
        return _parse_with_frontmatter(raw, m, rel_path)
    return _parse_plain(raw, rel_path, rel)


def _parse_with_frontmatter(raw: str, m: re.Match, rel_path: str) -> KnowledgeDoc:
    try:
        fm: dict[str, Any] = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        _fail(rel_path, f"YAML parse error: {e}")

    missing = REQUIRED_FIELDS - fm.keys()
    if missing:
        _fail(rel_path, f"missing required fields: {sorted(missing)}")

    title = fm["title"]
    if not isinstance(title, str) or not title.strip():
        _fail(rel_path, "title must be a non-empty string")

    category = fm["category"]
    if category not in VALID_CATEGORIES:
        _fail(rel_path, f"category must be one of {sorted(VALID_CATEGORIES)}, got {category!r}")

    tags = fm["tags"]
    if not isinstance(tags, list):
        _fail(rel_path, "tags must be a list")

    last_updated = str(fm["last_updated"])
    if isinstance(fm["last_updated"], date):
        last_updated = fm["last_updated"].isoformat()
    if not DATE_RE.match(last_updated):
        _fail(rel_path, f"last_updated must be YYYY-MM-DD, got {last_updated!r}")

    weight = float(fm.get("weight", 1.0))
    body = raw[m.end():]

    metadata = {
        "title": title, "category": category, "tags": tags,
        "last_updated": last_updated, "weight": weight,
    }
    return KnowledgeDoc(
        path=rel_path, title=title, category=category, tags=tags,
        last_updated=last_updated, weight=weight, body=body, metadata=metadata,
    )


def _parse_plain(raw: str, rel_path: str, rel: Path) -> KnowledgeDoc:
    """Infer metadata from file path and content for plain markdown files."""
    parts = rel.parts

    # Category from top-level directory
    top_dir = parts[0] if parts else ""
    category = _DIR_TO_CATEGORY.get(top_dir, "meta")

    # Tags from the subdirectory name (second level), normalized
    if len(parts) >= 2:
        sub = parts[1].replace("_", "-").lower()
        tags = [sub]
    else:
        tags = []

    # Title from first H1, falling back to filename stem
    h1 = H1_RE.search(raw)
    title = h1.group(1).strip() if h1 else rel.stem.replace("_", " ").replace("-", " ").title()

    last_updated = datetime.today().strftime("%Y-%m-%d")
    weight = 1.0

    metadata = {
        "title": title, "category": category, "tags": tags,
        "last_updated": last_updated, "weight": weight,
    }
    return KnowledgeDoc(
        path=rel_path, title=title, category=category, tags=tags,
        last_updated=last_updated, weight=weight, body=raw, metadata=metadata,
    )


def _fail(path: str, reason: str) -> None:
    print(f"ERROR: {path}: {reason}", file=sys.stderr)
    raise SystemExit(1)
