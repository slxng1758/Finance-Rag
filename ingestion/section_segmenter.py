"""Splits a 10-K/10-Q's extracted text into labeled Item sections.

SEC filings repeat their Item/Part headings at least twice: once in the table
of contents (tightly packed, little/no title text) and once at the real
section start. We first locate the *real* Part I/II/III/IV boundaries (Part
headings reliably have a large gap to the next heading, since each covers
many items, so the simple "largest gap" heuristic is safe for them). Once we
know where real Part I starts, everything before it is the TOC block by
construction — so item headings are then resolved by which real-Part region
they fall inside, not by gap size. This also correctly handles items with
*no* real body at all (e.g. "Item 6. [Reserved]"), which a pure gap heuristic
mishandles since a zero-body real section can have a smaller gap than its own
TOC entry.

It also resolves 10-Q's item-number reuse across Parts for free: Part I Item 1
("Financial Statements") and Part II Item 1 ("Legal Proceedings") are the same
key but fall in different regions, so no separate disambiguation is needed.
"""

import re
import warnings
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ITEM_RE = re.compile(r"^item\s+(\d{1,2}[a-c]?)\.?\s*[-–—]?\s*(.*)$", re.I)
PART_RE = re.compile(r"^part\s+(i{1,3}|iv)\b\s*[:.\-]?\s*(.*)$", re.I)
PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
BULLET_ONLY_RE = re.compile(r"^[•◦\-*]$")

# item_number -> role for 10-K, where Part doesn't disambiguate item numbers.
ROLE_MAP_10K = {
    "1": "business",
    "1A": "risk_factors",
    "1B": "unresolved_staff_comments",
    "1C": "cybersecurity",
    "2": "properties",
    "3": "legal_proceedings",
    "4": "mine_safety",
    "7": "mdna",
    "7A": "market_risk",
    "8": "financial_statements",
    "9A": "controls_and_procedures",
}

# (part, item_number) -> role for 10-Q, where the same item number means
# different things in Part I (financial info) vs Part II (other info).
ROLE_MAP_10Q = {
    ("I", "1"): "financial_statements",
    ("I", "2"): "mdna",
    ("I", "3"): "market_risk",
    ("I", "4"): "controls_and_procedures",
    ("II", "1"): "legal_proceedings",
    ("II", "1A"): "risk_factors",
    ("II", "2"): "unregistered_sales",
}


def get_role(form: str, part: str | None, item_number: str) -> str | None:
    if form == "10-K":
        return ROLE_MAP_10K.get(item_number)
    if form == "10-Q":
        return ROLE_MAP_10Q.get((part, item_number))
    return None


@dataclass
class Section:
    form: str
    part: str | None
    item_number: str
    role: str | None
    title: str
    paragraphs: list[str] = field(default_factory=list)
    match_quality: str = "ok"  # "ok" or "short" (e.g. "[Reserved]", incorporated by reference)

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


def _normalize_line(line: str) -> str:
    return line.replace("\xa0", " ").strip()


def extract_lines(html: str) -> list[str]:
    """HTML -> cleaned list of non-empty lines, with page-number/footer noise dropped."""
    soup = BeautifulSoup(html, "lxml")
    raw_text = soup.get_text(separator="\n")
    lines = []
    for raw in raw_text.split("\n"):
        line = _normalize_line(raw)
        if not line:
            continue
        if PAGE_NUM_RE.match(line):
            continue
        if line.lower() == "table of contents":
            continue
        if BULLET_ONLY_RE.match(line):
            continue
        lines.append(line)
    return lines


def _pick_real_by_gap(occurrences: list[tuple[int, str]]) -> dict[str, int]:
    """Pick, for each key, the occurrence with the largest gap to the next heading
    of any kind (ties go to the later occurrence). Safe for Part headings, which
    always cover substantial real content; not safe for items with empty real
    bodies, which is why items are resolved via region partitioning instead."""
    indices = [idx for idx, _ in occurrences]
    best: dict[str, tuple[int, int]] = {}
    for pos, (idx, key) in enumerate(occurrences):
        next_idx = indices[pos + 1] if pos + 1 < len(indices) else idx + 10_000
        gap = next_idx - idx
        if key not in best or gap >= best[key][0]:
            best[key] = (gap, idx)
    return {key: idx for key, (gap, idx) in best.items()}


def segment_filing(html: str, form: str) -> list[Section]:
    """Parse a filing's HTML into its real (non-TOC) Item sections, in document order."""
    lines = extract_lines(html)

    item_occurrences: list[tuple[int, str]] = []
    item_titles: dict[int, str] = {}
    part_occurrences: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        m = ITEM_RE.match(line)
        if m and len(line) < 200:
            item_num = m.group(1).upper()
            item_occurrences.append((i, item_num))
            item_titles[i] = m.group(2).strip() or item_num
            continue
        m = PART_RE.match(line)
        if m and len(line) < 100:
            part_occurrences.append((i, m.group(1).upper()))

    real_parts = _pick_real_by_gap(part_occurrences)
    regions = sorted(real_parts.items(), key=lambda kv: kv[1])  # [(part_name, start_idx), ...]

    if not regions:
        # Fallback for filings with no detectable Part headings: treat the whole
        # document as one region and resolve items by gap, same as Part headings.
        regions = [(None, 0)]

    sections: list[Section] = []
    for region_pos, (part_name, region_start) in enumerate(regions):
        region_end = regions[region_pos + 1][1] if region_pos + 1 < len(regions) else len(lines)

        in_region: dict[str, list[int]] = {}
        for idx, key in item_occurrences:
            if region_start <= idx < region_end:
                in_region.setdefault(key, []).append(idx)

        # Exactly one occurrence per key is the expected case (TOC is excluded by
        # construction, since it precedes the first real-Part boundary). If more
        # than one slipped into a single region, fall back to gap-based picking.
        chosen: list[tuple[str, int]] = []
        for key, idxs in in_region.items():
            if len(idxs) == 1:
                chosen.append((key, idxs[0]))
            else:
                sub = [(i, key) for i in idxs]
                picked = _pick_real_by_gap(sub)
                chosen.append((key, picked[key]))
        chosen.sort(key=lambda kv: kv[1])

        for pos, (item_num, start_idx) in enumerate(chosen):
            end_idx = chosen[pos + 1][1] if pos + 1 < len(chosen) else region_end
            body = lines[start_idx + 1 : end_idx]
            role = get_role(form, part_name, item_num)
            match_quality = "short" if sum(len(p) for p in body) < 200 else "ok"
            sections.append(
                Section(
                    form=form,
                    part=part_name,
                    item_number=item_num,
                    role=role,
                    title=item_titles[start_idx],
                    paragraphs=body,
                    match_quality=match_quality,
                )
            )

    # Already in document order: regions are processed in ascending start-index
    # order, and `chosen` within each region is sorted ascending too.
    return sections
