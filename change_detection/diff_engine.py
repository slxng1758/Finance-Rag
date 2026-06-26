"""Paragraph-level diffing within an aligned pair of sections.

Two passes:
1. Cheap pass — difflib.SequenceMatcher over paragraphs as opaque tokens.
   'equal' opcodes are discarded immediately at zero embedding/LLM cost; this
   typically eliminates 80-90%+ of a 10-K risk-factors section, since most of
   it is unchanged boilerplate year over year.
2. Reordering-safe pass — difflib's positional pairing for the *non*-equal
   opcodes is untrustworthy: a risk factor that moved from paragraph 3 to
   paragraph 7 shows up as a delete-at-3 plus an insert-at-7, which would be
   wrongly reported as "removed" + "added" if taken at face value. So instead
   we pool every paragraph touched by a non-equal opcode from both sides and
   solve a global similarity-based assignment between the two pools, which
   correctly matches a moved/reworded paragraph regardless of position drift.
"""

import difflib
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from embeddings import cosine_sim, embed

UNCHANGED_THRESHOLD = 0.92  # >= this: near-duplicate, drop (whitespace/punctuation noise)
MODIFIED_THRESHOLD = 0.55  # [MODIFIED_THRESHOLD, UNCHANGED_THRESHOLD): reworded/softened
# < MODIFIED_THRESHOLD or unmatched after assignment: genuinely new/dropped content

ChangeType = Literal["added", "removed", "modified"]


@dataclass
class ParagraphChange:
    change_type: ChangeType
    old_text: str | None
    new_text: str | None
    old_index: int | None  # index within the *old section's* paragraph list
    new_index: int | None  # index within the *new section's* paragraph list
    similarity: float | None
    reordered: bool = False


def _equal_ratio(a: str, b: str) -> float:
    """Cheap textual similarity used only to drop near-duplicate whitespace/
    punctuation noise that survived the paragraph split (no embedding needed)."""
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def diff_sections(old_paragraphs: list[str], new_paragraphs: list[str]) -> list[ParagraphChange]:
    """Diff two paragraph lists (from an already-aligned pair of sections)."""
    sm = difflib.SequenceMatcher(None, old_paragraphs, new_paragraphs, autojunk=False)

    old_pool_idx: list[int] = []  # indices into old_paragraphs touched by a non-equal opcode
    new_pool_idx: list[int] = []  # indices into new_paragraphs touched by a non-equal opcode

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        old_pool_idx.extend(range(i1, i2))
        new_pool_idx.extend(range(j1, j2))

    changes: list[ParagraphChange] = []

    if not old_pool_idx and not new_pool_idx:
        return changes

    old_pool = [old_paragraphs[i] for i in old_pool_idx]
    new_pool = [new_paragraphs[j] for j in new_pool_idx]

    if old_pool and new_pool:
        old_emb = embed(old_pool)
        new_emb = embed(new_pool)
        sims = cosine_sim(old_emb, new_emb)  # (n_old, n_new)
        row_ind, col_ind = linear_sum_assignment(-sims)  # maximize similarity

        matched_old, matched_new = set(), set()
        for rank, (r, c) in enumerate(zip(row_ind, col_ind)):
            sim = float(sims[r, c])
            matched_old.add(r)
            matched_new.add(c)
            if sim >= UNCHANGED_THRESHOLD:
                continue  # near-duplicate, drop — not a real change
            change_type: ChangeType = "modified" if sim >= MODIFIED_THRESHOLD else None
            if change_type is None:
                # Below the modified threshold even though linear_sum_assignment paired
                # them (it always produces min(n_old, n_new) pairs) — treat as independent
                # added/removed rather than a forced low-confidence "modified" pairing.
                changes.append(
                    ParagraphChange("removed", old_pool[r], None, old_pool_idx[r], None, None)
                )
                changes.append(
                    ParagraphChange("added", None, new_pool[c], None, new_pool_idx[c], None)
                )
                continue
            reordered = bool(r != c)  # position within the pool lists drifted -> moved, not edited in place
            changes.append(
                ParagraphChange(
                    "modified",
                    old_pool[r],
                    new_pool[c],
                    old_pool_idx[r],
                    new_pool_idx[c],
                    sim,
                    reordered=reordered,
                )
            )

        for r in range(len(old_pool)):
            if r not in matched_old:
                changes.append(ParagraphChange("removed", old_pool[r], None, old_pool_idx[r], None, None))
        for c in range(len(new_pool)):
            if c not in matched_new:
                changes.append(ParagraphChange("added", None, new_pool[c], None, new_pool_idx[c], None))
    elif old_pool:
        for r, text in enumerate(old_pool):
            changes.append(ParagraphChange("removed", text, None, old_pool_idx[r], None, None))
    else:
        for c, text in enumerate(new_pool):
            changes.append(ParagraphChange("added", None, text, None, new_pool_idx[c], None))

    return changes
