from ingestion.section_segmenter import segment_filing
from detect_changes import _diff_filing_pair

OLD_10K = """
<html><body>
<p>Part I</p><p>Item 1.</p><p>Item 1A.</p>
<p>Table of Contents</p>
<p>Part I</p>
<p>Item 1. Business</p>
<p>We design and manufacture things.</p>
<p>Item 1A. Risk Factors</p>
<p>Our business faces many risks.</p>
<p>Competition is intense in our industry.</p>
</body></html>
"""

NEW_10K = """
<html><body>
<p>Part I</p><p>Item 1.</p><p>Item 1A.</p>
<p>Table of Contents</p>
<p>Part I</p>
<p>Item 1. Business</p>
<p>We design and manufacture things and also services now.</p>
<p>Item 1A. Risk Factors</p>
<p>Our business faces many significant risks.</p>
<p>Competition is intense in our industry.</p>
<p>Regulatory risk has increased materially this year.</p>
</body></html>
"""


def test_diff_filing_pair_aligns_by_role_and_reports_paragraph_changes():
    old_sections = segment_filing(OLD_10K, "10-K")
    new_sections = segment_filing(NEW_10K, "10-K")
    report = _diff_filing_pair("old", old_sections, "new", new_sections)

    by_role = {s["role"]: s for s in report["sections"]}

    business = by_role["business"]
    assert business["match_method"] == "role"
    assert business["num_changes"] == 1
    assert business["changes"][0]["change_type"] == "modified"

    risk_factors = by_role["risk_factors"]
    assert risk_factors["num_changes"] == 1
    assert risk_factors["changes"][0]["change_type"] == "added"
    assert risk_factors["changes"][0]["new_text"] == "Regulatory risk has increased materially this year."


def test_diff_filing_pair_reports_added_section_when_item_only_in_new():
    old_sections = segment_filing(OLD_10K, "10-K")
    new_sections = segment_filing(NEW_10K, "10-K")
    new_sections.append(
        type(new_sections[0])(
            form="10-K", part="I", item_number="2", role="properties", title="Properties",
            paragraphs=["We lease office space."],
        )
    )
    report = _diff_filing_pair("old", old_sections, "new", new_sections)

    added = [s for s in report["sections"] if s["status"] == "added"]
    assert len(added) == 1
    assert added[0]["item"] == "2"
