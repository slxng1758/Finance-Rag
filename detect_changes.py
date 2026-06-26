"""CLI entry point that wires the change-detection pipeline together end to end:

    EDGAR fetch -> section segmentation -> section alignment -> paragraph diff

Run against live EDGAR data for a known company:
    python detect_changes.py NVDA --form 10-K --periods 3

Or run against two filings already on disk (no network call), e.g. the sample
HTML in data/filings/:
    python detect_changes.py --from-files data/filings/10k_1.html data/filings/10k_0.html --form 10-K
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from change_detection.diff_engine import diff_sections
from change_detection.section_aligner import align_sections
from ingestion.edgar_client import CIK_MAP, FilingRef, fetch_filing_html, list_filings
from ingestion.section_segmenter import Section, segment_filing

CACHE_DIR = Path("data/filings/cache")


def _cache_path(ref: FilingRef) -> Path:
    return CACHE_DIR / f"{ref.company}_{ref.form.replace('-', '')}_{ref.period_of_report}.html"


def _load_or_fetch(ref: FilingRef) -> str:
    """Fetch a filing's HTML, caching it to disk so re-runs (e.g. iterating on
    diff thresholds) don't re-hit EDGAR for the same accession."""
    path = _cache_path(ref)
    if path.exists():
        return path.read_text()
    html = fetch_filing_html(ref)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return html


def _diff_filing_pair(
    old_label: str, old_sections: list[Section], new_label: str, new_sections: list[Section]
) -> dict:
    """Align and diff one pair of filings, returning a JSON-serializable report."""
    section_reports = []
    for alignment in align_sections(old_sections, new_sections):
        if alignment.old_section is None or alignment.new_section is None:
            present = alignment.new_section or alignment.old_section
            section_reports.append(
                {
                    "status": "added" if alignment.old_section is None else "removed",
                    "item": present.item_number,
                    "title": present.title,
                }
            )
            continue

        changes = diff_sections(alignment.old_section.paragraphs, alignment.new_section.paragraphs)
        section_reports.append(
            {
                "status": "compared",
                "match_method": alignment.match_method,
                "match_similarity": alignment.similarity,
                "role": alignment.new_section.role or alignment.old_section.role,
                "old_item": alignment.old_section.item_number,
                "new_item": alignment.new_section.item_number,
                "title": alignment.new_section.title,
                "num_changes": len(changes),
                "changes": [asdict(c) for c in changes],
            }
        )

    return {"old_period": old_label, "new_period": new_label, "sections": section_reports}


def detect_changes(company: str, form: str, num_periods: int = 2) -> dict:
    """Fetch the `num_periods` most recent EDGAR filings of `form` for `company`
    and diff each consecutive pair (newest vs. next-newest, etc.)."""
    refs = list_filings(company, forms=(form,), limit_per_form=num_periods)
    if len(refs) < 2:
        raise ValueError(f"Need at least 2 filings to diff, found {len(refs)} for {company} {form}")

    parsed = [(ref, segment_filing(_load_or_fetch(ref), form)) for ref in refs]

    comparisons = []
    for (new_ref, new_sections), (old_ref, old_sections) in zip(parsed, parsed[1:]):
        old_label = f"{old_ref.period_of_report} ({old_ref.accession_number})"
        new_label = f"{new_ref.period_of_report} ({new_ref.accession_number})"
        comparisons.append(_diff_filing_pair(old_label, old_sections, new_label, new_sections))

    return {"company": company, "form": form, "comparisons": comparisons}


def detect_changes_from_files(old_path: Path, new_path: Path, form: str) -> dict:
    """Diff two already-downloaded filing HTML files directly, skipping EDGAR."""
    old_sections = segment_filing(old_path.read_text(), form)
    new_sections = segment_filing(new_path.read_text(), form)
    comparison = _diff_filing_pair(old_path.name, old_sections, new_path.name, new_sections)
    return {"company": None, "form": form, "comparisons": [comparison]}


def _print_summary(report: dict) -> None:
    label = report["company"] or "local files"
    print(f"\n=== {label} {report['form']} change detection ===")
    for comp in report["comparisons"]:
        print(f"\n{comp['old_period']} -> {comp['new_period']}")
        for sec in comp["sections"]:
            if sec["status"] in ("added", "removed"):
                print(f"  [{sec['status'].upper()}] Item {sec['item']} ({sec['title']})")
            elif sec["num_changes"] > 0:
                print(
                    f"  Item {sec['new_item']} ({sec['role'] or 'unmapped'}): "
                    f"{sec['num_changes']} paragraph changes [{sec['match_method']}]"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("company", nargs="?", choices=sorted(CIK_MAP), help="Ticker symbol (e.g. NVDA, JPM)")
    parser.add_argument("--form", choices=("10-K", "10-Q"), default="10-K")
    parser.add_argument("--periods", type=int, default=2, help="Number of most-recent filings to fetch and diff consecutively")
    parser.add_argument("--from-files", nargs=2, metavar=("OLD_HTML", "NEW_HTML"), help="Diff two local HTML files instead of fetching from EDGAR")
    parser.add_argument("--out", type=Path, help="Write the full JSON report to this path")
    args = parser.parse_args()

    if args.from_files:
        report = detect_changes_from_files(Path(args.from_files[0]), Path(args.from_files[1]), args.form)
    else:
        if not args.company:
            parser.error("company is required unless --from-files is given")
        report = detect_changes(args.company, args.form, args.periods)

    _print_summary(report)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
