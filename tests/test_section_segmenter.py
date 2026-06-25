from ingestion.section_segmenter import extract_lines, get_role, segment_filing

TOC_AND_BODY_10K = """
<html><body>
<p>Part I</p>
<p>Item 1.</p>
<p>Item 1A.</p>
<p>Part II</p>
<p>Item 5.</p>
<p>Item 6.</p>
<p>13</p>
<p>Table of Contents</p>
<p>Part I</p>
<p>Item 1. Business</p>
<p>We design and manufacture things.</p>
<p>Item 1A. Risk Factors</p>
<p>Our business faces many risks.</p>
<p>Another risk paragraph.</p>
<p>Part II</p>
<p>Item 5. Market for Common Equity</p>
<p>Our stock trades on an exchange.</p>
<p>Item 6. [Reserved]</p>
<p>Item 7. Management Discussion</p>
<p>We discuss results here.</p>
</body></html>
"""


def test_extract_lines_drops_page_numbers_and_toc_markers():
    lines = extract_lines(TOC_AND_BODY_10K)
    assert "13" not in lines
    assert "Table of Contents" not in lines


def test_segment_filing_skips_toc_and_finds_real_sections():
    sections = segment_filing(TOC_AND_BODY_10K, "10-K")
    by_item = {s.item_number: s for s in sections}

    assert by_item["1"].title == "Business"
    assert by_item["1"].paragraphs == ["We design and manufacture things."]

    assert by_item["1A"].title == "Risk Factors"
    assert by_item["1A"].paragraphs == [
        "Our business faces many risks.",
        "Another risk paragraph.",
    ]
    assert by_item["1A"].role == "risk_factors"


def test_segment_filing_handles_zero_body_reserved_item():
    sections = segment_filing(TOC_AND_BODY_10K, "10-K")
    by_item = {s.item_number: s for s in sections}
    assert by_item["6"].title == "[Reserved]"
    assert by_item["6"].paragraphs == []
    assert by_item["6"].match_quality == "short"


def test_segment_filing_assigns_correct_part():
    sections = segment_filing(TOC_AND_BODY_10K, "10-K")
    by_item = {s.item_number: s for s in sections}
    assert by_item["1"].part == "I"
    assert by_item["5"].part == "II"


def test_get_role_10k_is_keyed_by_item_number_only():
    assert get_role("10-K", "I", "1A") == "risk_factors"
    assert get_role("10-K", None, "7") == "mdna"


def test_get_role_10q_disambiguates_by_part():
    # Same item number ("1"/"2") means different things in Part I vs Part II of a 10-Q.
    assert get_role("10-Q", "I", "1") == "financial_statements"
    assert get_role("10-Q", "II", "1") == "legal_proceedings"
    assert get_role("10-Q", "I", "2") == "mdna"
    assert get_role("10-Q", "II", "2") == "unregistered_sales"
