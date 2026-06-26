# DisclosureFlow — SEC Filing Change Detector

Detects what actually changed between two SEC filings (10-K/10-Q) of the same
company, instead of leaving a reader to manually diff hundred-page documents.
Pulls real filings from SEC EDGAR, parses them into labeled disclosure
sections, aligns matching sections across filing periods (even across form
types — a 10-K's MD&A vs. a 10-Q's MD&A), and diffs paragraphs to surface
genuine additions/removals/rewording while ignoring unchanged boilerplate.

A secondary, optional layer (`ingest.py` / `app.py`) provides local RAG
Q&A over a folder of financial PDFs, for ad-hoc question answering.

## Quick start — change detection (core)

```bash
pip install -r requirements.txt

# Fetch live filings from SEC EDGAR and diff the two most recent 10-Ks
python detect_changes.py NVDA --form 10-K --periods 2

# Or run fully offline against filings already on disk
python detect_changes.py --from-files data/filings/10k_1.html data/filings/10k_0.html --form 10-K

# Write the full structured report to a file
python detect_changes.py NVDA --form 10-K --out report.json
```

Currently supports NVDA and JPM (see `CIK_MAP` in `ingestion/edgar_client.py`
to add more companies) and the 10-K/10-Q forms.

## How it works

```
ingestion/edgar_client.py        fetch filings from SEC EDGAR's official submissions API
        v
ingestion/section_segmenter.py   parse filing HTML into labeled Item sections
        v
change_detection/section_aligner.py   pair up "the same section" across two filings
        v
change_detection/diff_engine.py       diff paragraphs within each aligned pair
        v
detect_changes.py                CLI orchestrating the above end to end
```

- **Section parsing** distinguishes a filing's real sections from its
  duplicate table-of-contents entries, and correctly handles items with no
  real body (e.g. "Item 6. [Reserved]").
- **Section alignment** resolves matches in three tiers — semantic role
  (e.g. "mdna"), then (form, item number), then embedding similarity — so
  comparisons hold up even when an item is renumbered or a filing switches
  form type.
- **Paragraph diffing** is two-pass: a cheap exact-match pass
  (`difflib.SequenceMatcher`) drops unchanged paragraphs before any
  embedding model runs, then an optimal bipartite assignment
  (`scipy.optimize.linear_sum_assignment` over embedding cosine similarity)
  matches reworded or reordered paragraphs that a naive positional diff
  would misreport as unrelated additions/deletions.

## Evaluation

`section_aligner.py` and `diff_engine.py` both rely on hand-tuned similarity
thresholds, which unit tests can't validate on their own. `eval/score.py`
scores the pipeline's actual output against hand-labeled ground truth
(precision/recall per section):

```bash
python -m eval.score eval/ground_truth.json
```

`eval/ground_truth.json` ships with an empty `sections` list — see the
docstring in `eval/score.py` for how to label a section yourself.

## Testing

```bash
pytest
```

Covers section-parsing edge cases (TOC filtering, zero-body items, cross-form
item-number disambiguation) and an end-to-end alignment+diff integration test.

## Optional: local Q&A over PDFs

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.2

# Add PDFs to data/, then:
python ingest.py
streamlit run app.py
```

Indexes PDFs into a local Chroma vector store and answers questions via a
local LLM (no API keys, no API costs) with source citations.

## Project structure

```
ingestion/            EDGAR fetching + section parsing
change_detection/      section alignment + paragraph diffing
eval/                  precision/recall scoring against labeled ground truth
tests/                 pytest suite
detect_changes.py      CLI entry point for the change-detection pipeline
embeddings.py          shared embedding model (BAAI/bge-large-en-v1.5)
ingest.py / app.py     optional local PDF Q&A layer
data/filings/          sample SEC filing HTML for offline runs/tests
```
