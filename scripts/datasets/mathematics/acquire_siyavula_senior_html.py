"""Resumable, polite acquisition of Siyavula senior online textbook sections.

Reuses the retained Grade 8–9 parser and shared host lock. All records remain
unapproved and require source, rights, mathematics, curriculum and overlap review.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import urllib.error
import urllib.parse
import urllib.robotparser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

SCHOOL = Path("/mnt/codexa/pytorch/codexa-v3/school-math")
LEGACY = SCHOOL / "corpus-v1/workers/grades-8-9/extract.py"
BASE = "https://www.siyavula.com"
UA = "CodexaResearch/1.0 (educational corpus; contact research administrator)"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, data: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def load_legacy(root: Path) -> Any:
    spec = importlib.util.spec_from_file_location("siyavula_legacy_parser", LEGACY)
    if spec is None or spec.loader is None:
        raise ImportError("saved Siyavula parser not available")
    legacy: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    legacy.ROOT = root
    legacy.RAW = root / "raw"
    legacy.RAW.mkdir(parents=True, exist_ok=True)
    legacy.LOCK = SCHOOL / "corpus-v1/siyavula-host.lock"
    legacy.CACHE_PATH = root / "cache-hashes.json"
    legacy.CACHE = json.loads(legacy.CACHE_PATH.read_text()) if legacy.CACHE_PATH.exists() else {}
    return legacy


def parse_online_page(
    old: Any, grade: int, source: dict[str, Any], page: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Parse direct textbook section children; keep nested sections separate."""
    soup = BeautifulSoup(page, "html.parser")
    container = soup.select_one(".sv-book-section-view")
    if container is None:
        return [], "Online textbook container missing"
    sections = container.select("section.section")
    if not sections:
        return [], "No textbook sections (chapter navigation or unsupported page)"
    records: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        fragment = BeautifulSoup('<div id="content"></div>', "html.parser")
        content = fragment.select_one("#content")
        assert content is not None
        for child in section.children:
            if (
                isinstance(child, Tag)
                and child.name == "section"
                and "section" in (child.get("class") or [])
            ):
                continue
            content.append(copy.copy(child))
        rows, _ = old.parse_page(grade, source, str(fragment))
        section_id = str(section.get("id") or f"section-{index + 1:04d}")
        for row in rows:
            location = section_id + "/" + str(row["page_or_section"])
            row["page_or_section"] = location
            row["id"] = sha((source["source_id"] + "\0" + location).encode())
            row["source_section_id"] = section_id
            row["extraction_version"] = "senior-online-html-v2"
            records.append(row)
    return records, None if records else "Textbook sections had no extractable units"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    (root / "processed").mkdir(exist_ok=True)
    old = load_legacy(root)
    robots = old.cached(BASE + "/robots.txt", "robots.txt")
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(robots.decode("utf-8", errors="replace").splitlines())
    old.ROBOT = rp
    terms = old.cached(BASE + "/info/terms-and-conditions", "terms.html")
    if b"Creative Commons" not in terms:
        raise RuntimeError("Saved website terms lack expected Creative Commons notice")
    discovered = []
    grade_indices = []
    for grade in (10, 11, 12):
        url = f"{BASE}/read/za/mathematics/grade-{grade}"
        if not rp.can_fetch(UA, url):
            grade_indices.append({"grade": grade, "url": url, "status": "blocked_by_robots"})
            continue
        raw = old.cached(url, f"grade-{grade}-toc.html")
        soup = BeautifulSoup(raw, "html.parser")
        links: dict[str, str] = {}
        prefix = f"/read/za/mathematics/grade-{grade}/"
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href")
            if not isinstance(href, str):
                continue
            u = urllib.parse.urljoin(BASE, href)
            parts = urllib.parse.urlsplit(u)
            if (
                parts.scheme != "https"
                or parts.netloc != "www.siyavula.com"
                or not parts.path.startswith(prefix)
            ):
                continue
            clean = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
            if not rp.can_fetch(UA, clean):
                continue
            links.setdefault(clean, old.text(anchor))
        grade_indices.append(
            {
                "grade": grade,
                "url": url,
                "status": "discovered",
                "sha256": sha(raw),
                "unique_paths": len(links),
            }
        )
        for u, label in links.items():
            discovered.append(
                {"grade": grade, "url": u, "label": label, "source_id": old.sid(grade, u)}
            )
    queue = {
        "indices": grade_indices,
        "sections": discovered,
        "robots_sha256": sha(robots),
        "terms_sha256": sha(terms),
        "notice": "URLs are candidates; source terms, edition, fidelity and third-party rights remain under review.",
    }
    target = root / "queue.json"
    if target.exists():
        prior = json.loads(target.read_text())
        if prior != queue:
            raise ValueError(
                "discovery changed; preserve this queue and use a fresh output directory"
            )
    else:
        save(target, queue)
    print(json.dumps({"discovered": len(discovered), "indices": grade_indices}), flush=True)
    if args.discover_only:
        return
    all_counts = {"parsed": 0, "unparsed": 0, "rights_unresolved": 0, "failed": 0, "blocked": 0}
    for number, item in enumerate(discovered, 1):
        grade = item["grade"]
        url = item["url"]
        source_id = item["source_id"]
        destination = root / "processed" / (source_id + ".json")
        if destination.exists():
            existing = json.loads(destination.read_text())
            if not args.retry_failures or existing.get("status") not in (
                "failed",
                "blocked",
                "unparsed",
            ):
                continue
        if destination.exists():
            original = destination.read_bytes()
            history = root / "history"
            history.mkdir(exist_ok=True)
            archived = history / (source_id + "-" + sha(original) + ".json")
            if not archived.exists():
                archived.write_bytes(original)
        try:
            data = old.cached(url, source_id + ".html")
            page = data.decode("utf-8", errors="strict")
            digest = sha(data)
            if (
                "Creative Commons Attribution License" not in page
                or "creativecommons.org/licenses/by/3.0/" not in page
            ):
                result = {
                    **item,
                    "status": "rights_unresolved",
                    "source_sha256": digest,
                    "reason": "expected page-specific CC BY 3.0 notice missing; held",
                    "records": [],
                }
            else:
                src = {
                    "source_id": source_id,
                    "publisher": "Siyavula Education",
                    "title": f"Mathematics Grade {grade}: {item['label']}",
                    "edition": "Online South African textbook, undated",
                    "source_url": url,
                    "licence": "CC-BY-3.0",
                    "licence_location": url + " footer",
                    "attribution": "Siyavula Education, siyavula.com; CC BY 3.0; indicate changes",
                    "restrictions": "Exclude third-party material and trademark assets; exact use and website terms require review.",
                    "access_date": datetime.now(UTC).date().isoformat(),
                    "sha256": digest,
                    "raw_file": "raw/" + source_id + ".html",
                    "eligibility": "pending_independent_rights_and_overlap_review",
                }
                records, error = parse_online_page(old, grade, src, page)
                for record in records:
                    record["grade_band"] = "10-12"
                    record["extraction_worker"] = "senior-online"
                    record["rights_status"] = "pending_independent_rights_and_third_party_review"
                    record["curriculum_mapping_status"] = "pending_official_objective_review"
                    record["training_admission"] = "blocked"
                result = {
                    **item,
                    "status": "unparsed" if error else "parsed",
                    "source_sha256": digest,
                    "source": src,
                    "interactive_placeholder_present": "interactive_questions" in page,
                    "reason": error,
                    "records": records,
                }
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503):
                print(
                    json.dumps(
                        {
                            "stopped": str(exc),
                            "next_index": number,
                            "reason": "server rate or availability limit",
                        }
                    ),
                    flush=True,
                )
                raise
            result = {
                **item,
                "status": "blocked" if exc.code == 403 else "failed",
                "reason": str(exc),
                "records": [],
            }
        except (OSError, UnicodeError, ValueError, urllib.error.URLError) as exc:
            result = {**item, "status": "failed", "reason": str(exc), "records": []}
        save(destination, result)
        all_counts[result["status"]] += 1
        print(
            json.dumps(
                {
                    "index": number,
                    "total": len(discovered),
                    "grade": grade,
                    "status": result["status"],
                    "records": len(result["records"]),
                }
            ),
            flush=True,
        )
    print(
        json.dumps({"run_new_page_counts": all_counts, "status": "all extracted records held"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
