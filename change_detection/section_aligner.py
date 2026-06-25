"""Aligns "the same logical section" across two filings of a company from
different periods, so the diff engine always compares like with like.

Resolution order:
1. role match — handles cross-form comparisons (10-K Item 7 <-> 10-Q Item 2,
   both "mdna") and same-form comparisons where role is known.
2. normalized (form, item_number) match — fallback when role is unmapped
   (e.g. Item 9B "Other Information" has no named role) but both filings are
   the same form type and item number.
3. embedding-similarity bipartite fallback over title+lead-text, only for
   sections that resolved neither of the above. Below cosine 0.55 a section
   is left `unmatched` rather than forced into a bad pairing — that itself is
   a reportable signal ("section added/removed/restructured").
"""

from dataclasses import dataclass

import numpy as np

from embeddings import cosine_sim, embed
from ingestion.section_segmenter import Section

EMBEDDING_FALLBACK_THRESHOLD = 0.55


@dataclass
class SectionAlignment:
    old_section: Section | None
    new_section: Section | None
    match_method: str  # "role", "item_number", "embedding_fallback", "unmatched"
    similarity: float | None = None


def _lead_text(section: Section, n_chars: int = 200) -> str:
    return (section.title + " " + section.text)[:n_chars]


def align_sections(
    old_sections: list[Section], new_sections: list[Section]
) -> list[SectionAlignment]:
    """Pair up sections from an older and a newer filing of the same company."""
    old_remaining = list(old_sections)
    new_remaining = list(new_sections)
    alignments: list[SectionAlignment] = []

    # 1. Role match (covers cross-form comparisons and is the most semantically
    # reliable signal we have, since it's keyed off SEC's own Item taxonomy).
    old_by_role: dict[str, list[Section]] = {}
    for s in old_remaining:
        if s.role:
            old_by_role.setdefault(s.role, []).append(s)
    new_by_role: dict[str, list[Section]] = {}
    for s in new_remaining:
        if s.role:
            new_by_role.setdefault(s.role, []).append(s)

    for role in list(old_by_role):
        if role in new_by_role and old_by_role[role] and new_by_role[role]:
            old_sec = old_by_role[role].pop(0)
            new_sec = new_by_role[role].pop(0)
            alignments.append(SectionAlignment(old_sec, new_sec, "role", similarity=1.0))
            old_remaining.remove(old_sec)
            new_remaining.remove(new_sec)

    # 2. Normalized (form, item_number) match for whatever has no role mapping,
    # restricted to same-form pairs (cross-form item numbers aren't comparable
    # without a role, e.g. 10-K Item 2 "Properties" vs 10-Q Item 2 "MD&A").
    old_by_item: dict[tuple[str, str], list[Section]] = {}
    for s in old_remaining:
        old_by_item.setdefault((s.form, s.item_number), []).append(s)
    new_by_item: dict[tuple[str, str], list[Section]] = {}
    for s in new_remaining:
        new_by_item.setdefault((s.form, s.item_number), []).append(s)

    for key in list(old_by_item):
        if key in new_by_item and old_by_item[key] and new_by_item[key]:
            old_sec = old_by_item[key].pop(0)
            new_sec = new_by_item[key].pop(0)
            alignments.append(SectionAlignment(old_sec, new_sec, "item_number", similarity=1.0))
            old_remaining.remove(old_sec)
            new_remaining.remove(new_sec)

    # 3. Embedding-similarity bipartite fallback for whatever's left.
    if old_remaining and new_remaining:
        old_emb = embed([_lead_text(s) for s in old_remaining])
        new_emb = embed([_lead_text(s) for s in new_remaining])
        sims = cosine_sim(old_emb, new_emb)  # (n_old, n_new)

        pairs = []
        for i in range(sims.shape[0]):
            for j in range(sims.shape[1]):
                pairs.append((sims[i, j], i, j))
        pairs.sort(reverse=True)

        used_old, used_new = set(), set()
        for sim, i, j in pairs:
            if i in used_old or j in used_new:
                continue
            if sim < EMBEDDING_FALLBACK_THRESHOLD:
                continue
            alignments.append(
                SectionAlignment(
                    old_remaining[i], new_remaining[j], "embedding_fallback", similarity=float(sim)
                )
            )
            used_old.add(i)
            used_new.add(j)

        old_remaining = [s for i, s in enumerate(old_remaining) if i not in used_old]
        new_remaining = [s for j, s in enumerate(new_remaining) if j not in used_new]

    # 4. Whatever's left is genuinely unmatched — section added or removed.
    for s in old_remaining:
        alignments.append(SectionAlignment(s, None, "unmatched"))
    for s in new_remaining:
        alignments.append(SectionAlignment(None, s, "unmatched"))

    return alignments
