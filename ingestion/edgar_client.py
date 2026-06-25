"""Fetches 10-K/10-Q filings directly from SEC EDGAR (free, no API key —
just a required identifying User-Agent header per SEC's fair-access policy).

We use EDGAR's `submissions` JSON API (gives periodOfReport/accession/form per
filing, so period identity never depends on parsing filenames or calendar-year
strings) and fetch the primary document as HTML, which preserves heading
structure that PDF-to-text extraction would lose.
"""

import os
import time
from dataclasses import dataclass

import requests

EDGAR_USER_AGENT = os.environ.get(
    "EDGAR_USER_AGENT", "Finance-Disclosure-Change-Detector sarahling1758@gmail.com"
)
HEADERS = {"User-Agent": EDGAR_USER_AGENT}

# Hardcoded CIKs for the two companies in scope (NVIDIA = primary corpus,
# JPMorgan = different-sector generalization smoke test). EDGAR also publishes
# a full ticker->CIK map at https://www.sec.gov/files/company_tickers.json if
# this needs to be extended to more companies later.
CIK_MAP = {
    "NVDA": "0001045810",
    "JPM": "0000019617",
}

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_doc}"


@dataclass
class FilingRef:
    company: str
    cik: str
    form: str  # "10-K" or "10-Q"
    accession_number: str  # e.g. "0001045810-24-000029"
    period_of_report: str  # e.g. "2024-01-28" — the actual fiscal period end, not filing date
    filing_date: str
    primary_document: str

    @property
    def period_id(self) -> str:
        """Stable identifier for this filing, independent of filename conventions."""
        return f"{self.company}_{self.form}_{self.period_of_report}"


def _get(url: str) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(0.15)  # stay well under SEC's 10 req/sec rate limit
    return resp


def _refs_from_block(block: dict, company: str, cik: str, forms: tuple[str, ...]) -> list[FilingRef]:
    out = []
    for i in range(len(block["form"])):
        form = block["form"][i]
        if form not in forms:
            continue
        out.append(
            FilingRef(
                company=company,
                cik=cik,
                form=form,
                accession_number=block["accessionNumber"][i],
                period_of_report=block["reportDate"][i],
                filing_date=block["filingDate"][i],
                primary_document=block["primaryDocument"][i],
            )
        )
    return out


def list_filings(
    company: str, forms=("10-K", "10-Q"), limit_per_form: int = 4, max_pages: int = 10
) -> list[FilingRef]:
    """Return the most recent `limit_per_form` filings of each form type for `company`,
    sorted newest-first within each form, using SEC's submissions API.

    For filers with very high overall filing volume (e.g. large banks with constant
    Form 4 insider-trading activity), the form we want can roll off the "recent" window
    within a few months — `data["filings"]["files"]` lists older paginated shards, which
    we walk through (oldest-filed-first ordering within `files`, so page 1 is the
    next-oldest chunk after "recent") until each requested form has enough hits or we
    run out of pages."""
    cik = CIK_MAP[company]
    data = _get(SUBMISSIONS_URL.format(cik=cik)).json()

    by_form: dict[str, list[FilingRef]] = {f: [] for f in forms}
    for ref in _refs_from_block(data["filings"]["recent"], company, cik, forms):
        by_form[ref.form].append(ref)

    pages = data["filings"].get("files", [])
    for page in pages[:max_pages]:
        if all(len(by_form[f]) >= limit_per_form for f in forms):
            break
        page_data = _get(SUBMISSIONS_PAGE_URL.format(name=page["name"])).json()
        for ref in _refs_from_block(page_data, company, cik, forms):
            by_form[ref.form].append(ref)

    out: list[FilingRef] = []
    for f in forms:
        out.extend(by_form[f][:limit_per_form])
    return out


def fetch_filing_html(ref: FilingRef) -> str:
    """Download the raw HTML of a filing's primary document."""
    accession_nodash = ref.accession_number.replace("-", "")
    cik_int = str(int(ref.cik))
    url = DOC_URL.format(cik_int=cik_int, accession_nodash=accession_nodash, primary_doc=ref.primary_document)
    return _get(url).text
