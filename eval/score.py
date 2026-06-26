"""Scores the change-detection pipeline (section_aligner + diff_engine) against
hand-labeled ground truth.

Both modules make embedding-similarity judgment calls using fixed thresholds
(see UNCHANGED_THRESHOLD / MODIFIED_THRESHOLD in diff_engine.py and
EMBEDDING_FALLBACK_THRESHOLD in section_aligner.py). Unit tests can confirm the
code runs without crashing, but only a human-labeled "here's what actually
changed in this real filing pair" example can confirm those thresholds are
catching genuine changes and ignoring boilerplate.

Run:
    python -m eval.score eval/ground_truth.json

To add a labeled example, edit ground_truth.json: pick a section (by role)
from a real filing pair already in data/filings/, read its old vs. new
paragraphs yourself, and list every change a careful human reviewer would
expect diff_sections() to find — change_type plus old_text/new_text copied
exactly as they appear in the document. Only sections you've actually
reviewed should appear in "sections"; score_file only scores what's there.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from change_detection.diff_engine import diff_sections
from change_detection.section_aligner import align_sections
from ingestion.section_segmenter import segment_filing


def _norm(text: str | None) -> str | None:
    return text.strip() if text else text


def _change_matches(predicted: dict, expected: dict) -> bool:
    if predicted["change_type"] != expected["change_type"]:
        return False
    for key in ("old_text", "new_text"):
        exp_val = expected.get(key)
        if exp_val is not None and _norm(predicted.get(key)) != _norm(exp_val):
            return False
    return True


def _score_section(predicted_changes: list[dict], expected_changes: list[dict]) -> dict:
    """Greedy one-to-one matching between predicted and expected changes."""
    matched_expected: set[int] = set()
    matched_predicted: set[int] = set()
    for p_idx, predicted in enumerate(predicted_changes):
        for e_idx, expected in enumerate(expected_changes):
            if e_idx in matched_expected:
                continue
            if _change_matches(predicted, expected):
                matched_expected.add(e_idx)
                matched_predicted.add(p_idx)
                break

    true_positives = len(matched_predicted)
    precision = true_positives / len(predicted_changes) if predicted_changes else 1.0
    recall = true_positives / len(expected_changes) if expected_changes else 1.0
    return {
        "true_positives": true_positives,
        "num_predicted": len(predicted_changes),
        "num_expected": len(expected_changes),
        "precision": precision,
        "recall": recall,
    }


def score_file(ground_truth_path: Path) -> dict:
    gt = json.loads(ground_truth_path.read_text())
    form = gt["form"]
    old_html = Path(gt["old_file"]).read_text()
    new_html = Path(gt["new_file"]).read_text()

    old_sections = segment_filing(old_html, form)
    new_sections = segment_filing(new_html, form)
    alignments_by_role = {
        (a.new_section or a.old_section).role: a
        for a in align_sections(old_sections, new_sections)
        if a.old_section and a.new_section and (a.old_section.role or a.new_section.role)
    }

    section_results = []
    for section_gt in gt.get("sections", []):
        role = section_gt["role"]
        expected_changes = section_gt["expected_changes"]
        alignment = alignments_by_role.get(role)
        if alignment is None:
            section_results.append(
                {"role": role, "error": "section not aligned by pipeline", **_score_section([], expected_changes)}
            )
            continue
        predicted_changes = [
            asdict(c) for c in diff_sections(alignment.old_section.paragraphs, alignment.new_section.paragraphs)
        ]
        section_results.append({"role": role, **_score_section(predicted_changes, expected_changes)})

    return {"company": gt.get("company"), "form": form, "sections": section_results}


def _print_report(report: dict) -> None:
    sections = report["sections"]
    if not sections:
        print("No labeled sections in ground truth yet — see eval/score.py docstring for how to add them.")
        return

    print(f"=== Eval: {report.get('company') or 'unknown'} {report['form']} ===")
    total_tp = total_pred = total_exp = 0
    for s in sections:
        flag = f" [{s['error']}]" if "error" in s else ""
        print(
            f"  {s['role']:<28} precision={s['precision']:.2f}  recall={s['recall']:.2f}  "
            f"(tp={s['true_positives']}, predicted={s['num_predicted']}, expected={s['num_expected']}){flag}"
        )
        total_tp += s["true_positives"]
        total_pred += s["num_predicted"]
        total_exp += s["num_expected"]

    overall_precision = total_tp / total_pred if total_pred else 1.0
    overall_recall = total_tp / total_exp if total_exp else 1.0
    print(f"\nOverall: precision={overall_precision:.2f}  recall={overall_recall:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ground_truth", type=Path, nargs="?", default=Path("eval/ground_truth.json"))
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(f"Ground truth file not found: {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    report = score_file(args.ground_truth)
    _print_report(report)


if __name__ == "__main__":
    main()
