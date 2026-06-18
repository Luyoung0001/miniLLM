#!/usr/bin/env python3

from __future__ import annotations

import re
import shutil
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[2]
COURSE_ROOT = REPO_ROOT / "course"
SITE_ROOT = REPO_ROOT / "course-site"
COLLECTION_ROOT = SITE_ROOT / "pages" / "_course"
DATA_ROOT = SITE_ROOT / "_data"
REPO_WEB_BASE = "https://github.com/Luyoung0001/miniLLM"
REPO_DEFAULT_BRANCH = "main"
LINK_PATTERN = re.compile(r"(?P<prefix>!?\[[^\]]+\]\()(?P<target>[^)]+)(?P<suffix>\))")

MAIN_SEQUENCE = [
    "README.md",
    "00-orientation.md",
    "practice/README.md",
    "chapters/ch01-step0-tensor.md",
    "chapters/ch02-step1-tokenizer.md",
    "chapters/ch03-step2-embedding.md",
    "chapters/ch04-step3-attention.md",
    "chapters/ch05-step4-transformer.md",
    "chapters/ch06-step5-gpt.md",
    "chapters/ch07-step6-training.md",
    "chapters/ch08-step7-generation.md",
    "chapters/ch09-step8-chat.md",
    "chapters/ch10-step9-http.md",
    "chapters/ch11-step10-kv-cache.md",
    "chapters/ch12-step11-bpe.md",
    "chapters/ch13-end-to-end.md",
]

APPENDIX_SEQUENCE = [
    "appendices/A1-prerequisites.md",
    "appendices/A2-pitfalls.md",
    "appendices/A3-references.md",
]

MAINTAINER_SEQUENCE = [
    "00-architecture.md",
    "appendices/A4-practice-submodule-migration.md",
]

EXCLUDED_SOURCE_PAGES = {
    "00-architecture.md",
    "appendices/A4-practice-submodule-migration.md",
    "practice/labs/lab11-step10/solution/README.md",
}


def build_course_basename_index() -> dict[str, list[PurePosixPath]]:
    index: dict[str, list[PurePosixPath]] = {}
    for path in COURSE_ROOT.rglob("*.md"):
        rel = PurePosixPath(path.relative_to(COURSE_ROOT).as_posix())
        index.setdefault(path.name, []).append(rel)
    return index


COURSE_BASENAME_INDEX = build_course_basename_index()


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return {}, text
    raw_meta = parts[1]
    body = parts[2]
    meta: dict[str, str] = {}
    for line in raw_meta.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, body


def extract_first_h1(body: str) -> tuple[str | None, str]:
    lines = body.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            rest = "\n".join(lines[:idx] + lines[idx + 1 :]).lstrip("\n")
            return title, rest
    return None, body


def choose_title(meta: dict[str, str], first_h1: str | None, src_rel: PurePosixPath) -> str:
    meta_title = meta.get("title")
    if first_h1 and (not meta_title or meta_title == src_rel.stem):
        return first_h1
    return meta_title or src_rel.stem


def quote_yaml(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def target_relpath(src_rel: PurePosixPath) -> PurePosixPath:
    if src_rel.name in {"README.md", "TASK.md"}:
        if str(src_rel.parent) == ".":
            return PurePosixPath("index.md")
        return src_rel.parent / "index.md"
    return src_rel


def url_for_src(src_rel: str) -> str:
    target = target_relpath(PurePosixPath(src_rel))
    if target.name == "index.md":
        if str(target.parent) == ".":
            return "/course/"
        return f"/course/{target.parent.as_posix()}/"
    return f"/course/{target.with_suffix('').as_posix()}/"


def repo_url_for_path(repo_rel: PurePosixPath, is_dir: bool) -> str:
    kind = "tree" if is_dir else "blob"
    return f"{REPO_WEB_BASE}/{kind}/{REPO_DEFAULT_BRANCH}/{repo_rel.as_posix()}"


def split_link_target(target: str) -> tuple[str, str]:
    stripped = target.strip()
    if " " not in stripped:
        return stripped, ""
    path_part, suffix = stripped.split(" ", 1)
    return path_part, f" {suffix}"


def split_fragment(target: str) -> tuple[str, str]:
    if "#" not in target:
        return target, ""
    path_part, fragment = target.split("#", 1)
    return path_part, f"#{fragment}"


def resolve_course_doc(abs_target: Path) -> PurePosixPath | None:
    try:
        course_rel = abs_target.relative_to(COURSE_ROOT)
    except ValueError:
        return None

    rel_posix = PurePosixPath(course_rel.as_posix())
    if abs_target.is_dir():
        for candidate in ("README.md", "TASK.md"):
            candidate_path = abs_target / candidate
            if candidate_path.exists():
                return rel_posix / candidate
        return None

    if abs_target.exists() and abs_target.suffix.lower() == ".md":
        return rel_posix

    return None


def fallback_course_doc(path_part: str) -> PurePosixPath | None:
    basename = PurePosixPath(path_part).name
    matches = COURSE_BASENAME_INDEX.get(basename, [])
    if len(matches) == 1:
        return matches[0]
    return None


def repo_url_from_abs_target(abs_target: Path) -> str | None:
    if not abs_target.exists():
        return None
    try:
        repo_rel = abs_target.relative_to(REPO_ROOT)
    except ValueError:
        return None
    return repo_url_for_path(PurePosixPath(repo_rel.as_posix()), abs_target.is_dir())


def fallback_repo_url(path_part: str) -> str | None:
    basename = PurePosixPath(path_part.rstrip("/")).name
    if not basename:
        return None
    candidate = REPO_ROOT / basename
    if candidate.exists():
        return repo_url_for_path(PurePosixPath(basename), candidate.is_dir())
    return None


def rewrite_link_target(src_rel: PurePosixPath, target: str) -> str:
    path_part, title_suffix = split_link_target(target)
    if not path_part or path_part.startswith(("http://", "https://", "mailto:", "#", "/")):
        return target

    raw_path, fragment = split_fragment(path_part)
    base_dir = (COURSE_ROOT / src_rel.parent).resolve()
    abs_target = (base_dir / raw_path).resolve()

    course_doc = resolve_course_doc(abs_target)
    if course_doc is not None:
        return f"{url_for_src(course_doc.as_posix())}{fragment}{title_suffix}"

    course_doc = fallback_course_doc(raw_path)
    if course_doc is not None:
        return f"{url_for_src(course_doc.as_posix())}{fragment}{title_suffix}"

    repo_url = repo_url_from_abs_target(abs_target)
    if repo_url is not None:
        return f"{repo_url}{fragment}{title_suffix}"

    repo_url = fallback_repo_url(raw_path)
    if repo_url is not None:
        return f"{repo_url}{fragment}{title_suffix}"

    return target


def rewrite_links(body: str, src_rel: PurePosixPath) -> str:
    def replace(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        target = match.group("target")
        suffix = match.group("suffix")
        return f"{prefix}{rewrite_link_target(src_rel, target)}{suffix}"

    return LINK_PATTERN.sub(replace, body)


def source_pages() -> list[PurePosixPath]:
    return sorted(
        PurePosixPath(path.relative_to(COURSE_ROOT).as_posix())
        for path in COURSE_ROOT.rglob("*.md")
        if path.relative_to(COURSE_ROOT).as_posix() not in EXCLUDED_SOURCE_PAGES
    )


def build_prev_next_map(title_map: dict[str, str]) -> dict[str, dict[str, dict[str, str]]]:
    mapping: dict[str, dict[str, dict[str, str]]] = {}
    for idx, src in enumerate(MAIN_SEQUENCE):
        entry: dict[str, dict[str, str]] = {}
        if idx > 0:
            prev_src = MAIN_SEQUENCE[idx - 1]
            entry["prev_page"] = {
                "title": title_map[prev_src],
                "url": url_for_src(prev_src),
            }
        if idx < len(MAIN_SEQUENCE) - 1:
            next_src = MAIN_SEQUENCE[idx + 1]
            entry["next_page"] = {
                "title": title_map[next_src],
                "url": url_for_src(next_src),
            }
        mapping[src] = entry
    return mapping


def format_front_matter(
    meta: dict[str, str],
    source_path: str,
    permalink: str,
    nav_meta: dict[str, dict[str, str]],
) -> str:
    lines = ["---"]
    preferred_order = ["title", "layout", "permalink", "lab", "step", "hours", "deliverable", "type"]
    seen = set()
    merged = dict(meta)
    merged["layout"] = "course_page"
    merged["permalink"] = permalink
    for key in preferred_order:
        if key in merged:
            lines.append(f"{key}: {quote_yaml(merged[key])}")
            seen.add(key)
    for key in sorted(merged):
        if key in seen:
            continue
        lines.append(f"{key}: {quote_yaml(merged[key])}")
    for key in ("prev_page", "next_page"):
        if key in nav_meta:
            lines.append(f"{key}:")
            lines.append(f"  title: {quote_yaml(nav_meta[key]['title'])}")
            lines.append(f"  url: {quote_yaml(nav_meta[key]['url'])}")
    lines.append(f"source_path: {quote_yaml(source_path)}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def write_page(src_rel: PurePosixPath, title_map: dict[str, str], prev_next_map: dict[str, dict[str, dict[str, str]]]) -> None:
    src_path = COURSE_ROOT / src_rel
    raw = src_path.read_text(encoding="utf-8")
    meta, body = split_front_matter(raw)
    first_h1, body = extract_first_h1(body)
    src_key = src_rel.as_posix()
    meta["title"] = choose_title(meta, first_h1, src_rel)
    body = rewrite_links(body, src_rel)
    target = COLLECTION_ROOT / Path(target_relpath(src_rel))
    target.parent.mkdir(parents=True, exist_ok=True)
    front_matter = format_front_matter(
        meta,
        f"course/{src_key}",
        url_for_src(src_key),
        prev_next_map.get(src_key, {}),
    )
    target.write_text(front_matter + body.rstrip() + "\n", encoding="utf-8")
    title_map[src_key] = meta["title"]


def build_navigation(title_map: dict[str, str]) -> str:
    def nav_children(items: list[str], anchors: bool = False) -> list[str]:
        lines: list[str] = []
        for item in items:
            title = title_map[item]
            url = url_for_src(item)
            lines.append(f"      - title: {quote_yaml(title)}")
            lines.append(f"        url: {quote_yaml(url)}")
        return lines

    continuous_items = [
        ("course-home", title_map["README.md"]),
        ("chapter-0", title_map["00-orientation.md"]),
        ("lab-guide", title_map["practice/README.md"]),
    ] + [
        (PurePosixPath(src).stem, title_map[src])
        for src in MAIN_SEQUENCE[3:]
    ]

    lines = [
        "main:",
        f"  - title: {quote_yaml('课程首页')}",
        f"    url: {quote_yaml('/course/')}",
        f"  - title: {quote_yaml('连续阅读')}",
        f"    url: {quote_yaml('/course/continuous/')}",
        f"  - title: {quote_yaml('Lab 说明')}",
        f"    url: {quote_yaml('/course/practice/')}",
        "",
        "course:",
        f"  - title: {quote_yaml('课程入口')}",
        "    children:",
        f"      - title: {quote_yaml(title_map['README.md'])}",
        f"        url: {quote_yaml(url_for_src('README.md'))}",
        f"      - title: {quote_yaml(title_map['00-orientation.md'])}",
        f"        url: {quote_yaml(url_for_src('00-orientation.md'))}",
        f"      - title: {quote_yaml(title_map['practice/README.md'])}",
        f"        url: {quote_yaml(url_for_src('practice/README.md'))}",
        f"  - title: {quote_yaml('主线章节')}",
        "    children:",
    ]
    lines.extend(nav_children(MAIN_SEQUENCE[3:]))
    lines.extend(
        [
            f"  - title: {quote_yaml('附录')}",
            "    children:",
        ]
    )
    lines.extend(nav_children(APPENDIX_SEQUENCE))
    lines.extend(
        [
            "",
            "course-continuous:",
            f"  - title: {quote_yaml('连续阅读')}",
            "    children:",
        ]
    )
    for anchor, title in continuous_items:
        lines.append(f"      - title: {quote_yaml(title)}")
        lines.append(f"        url: {quote_yaml(f'/course/continuous/#{anchor}')}")
    return "\n".join(lines) + "\n"


def build_continuous(title_map: dict[str, str]) -> str:
    entries = [
        ("course-home", "README.md"),
        ("chapter-0", "00-orientation.md"),
        ("lab-guide", "practice/README.md"),
    ] + [(PurePosixPath(src).stem, src) for src in MAIN_SEQUENCE[3:]]

    lines = [
        "---",
        f"title: {quote_yaml('miniLLM 课程连续阅读')}",
        "layout: course_page",
        "permalink: '/course/continuous/'",
        "type: '连续阅读'",
        "sidebar:",
        "  nav: course-continuous",
        "source_path: 'generated:continuous'",
        "---",
        "",
    ]
    for anchor, src in entries:
        source_path = f"course/{src}"
        lines.extend(
            [
                "{% for page in site.course %}",
                f"  {{% if page.source_path == \"{source_path}\" %}}",
                f"<section id=\"{anchor}\" class=\"chapter-section\" markdown=\"1\">",
                f"## {{{{ page.title }}}}",
                "{{ page.content }}",
                "</section>",
                "  {% endif %}",
                "{% endfor %}",
                "",
            ]
        )
    lines.extend(
        [
            "<script>",
            "document.addEventListener('DOMContentLoaded', function() {",
            "  const sections = document.querySelectorAll('.chapter-section');",
            "  const navLinks = document.querySelectorAll('.sidebar a[href*=\"#\"]');",
            "  function updateActiveLink() {",
            "    let current = '';",
            "    sections.forEach(section => {",
            "      const sectionTop = section.offsetTop;",
            "      if (window.pageYOffset >= sectionTop - 120) {",
            "        current = section.getAttribute('id');",
            "      }",
            "    });",
            "    navLinks.forEach(link => {",
            "      link.classList.remove('active');",
            "      if (link.getAttribute('href').includes('#' + current)) {",
            "        link.classList.add('active');",
            "      }",
            "    });",
            "  }",
            "  window.addEventListener('scroll', updateActiveLink);",
            "  updateActiveLink();",
            "});",
            "</script>",
            "",
        ]
    )
    return "\n".join(lines)


def reset_generated_content() -> None:
    if COLLECTION_ROOT.exists():
        shutil.rmtree(COLLECTION_ROOT)
    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    COLLECTION_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    reset_generated_content()
    title_map: dict[str, str] = {}

    raw_pages = source_pages()
    for src_rel in raw_pages:
        raw = (COURSE_ROOT / src_rel).read_text(encoding="utf-8")
        meta, body = split_front_matter(raw)
        first_h1, _ = extract_first_h1(body)
        title_map[src_rel.as_posix()] = choose_title(meta, first_h1, src_rel)

    prev_next_map = build_prev_next_map(title_map)

    for src_rel in raw_pages:
        write_page(src_rel, title_map, prev_next_map)

    (DATA_ROOT / "navigation.yml").write_text(build_navigation(title_map), encoding="utf-8")
    (COLLECTION_ROOT / "continuous.md").write_text(build_continuous(title_map), encoding="utf-8")

    print("Synced course markdown into course-site/")


if __name__ == "__main__":
    main()
