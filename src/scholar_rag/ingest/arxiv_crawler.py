"""
arXiv crawler — fetches paper metadata via the arXiv API and optionally
downloads PDFs.  Designed to be resumable: already-downloaded entries are
skipped on re-run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator
from urllib.parse import urlencode

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from xml.etree import ElementTree as ET


ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str          # ISO-8601 date string
    updated: str
    pdf_url: str
    pdf_path: str | None = None
    doi: str | None = None
    journal_ref: str | None = None
    comments: str | None = None
    extra: dict = field(default_factory=dict)


class ArxivCrawler:
    """
    Fetch papers from the arXiv API for one or more category+date combinations.

    Usage
    -----
    crawler = ArxivCrawler(raw_dir=Path("data/raw"), rate_limit_s=3.0)
    for paper in crawler.fetch(categories=["cs.CL"], max_results=500):
        crawler.download_pdf(paper)
    """

    def __init__(self, raw_dir: Path, rate_limit_s: float = 3.0) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_s = rate_limit_s
        self._client = httpx.Client(timeout=60.0, follow_redirects=True)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fetch(
        self,
        categories: list[str],
        max_results: int = 500,
        date_from: str | None = None,
        start: int = 0,
        query_terms: list[str] | None = None,
    ) -> Iterator[ArxivPaper]:
        """
        Yield ArxivPaper objects, paginating automatically.

        query_terms: optional list of topic keywords/phrases. When provided,
        results are restricted to papers whose title or abstract matches ANY of
        the terms (OR-combined), in addition to the category filter. Use this to
        target a sub-topic — e.g. ["large language model", "RAG",
        "retrieval-augmented"] — within broad categories like cs.LG.
        """
        query = self._build_query(categories, date_from, query_terms)
        batch_size = min(100, max_results)
        fetched = 0

        while fetched < max_results:
            params = {
                "search_query": query,
                "start": start + fetched,
                "max_results": min(batch_size, max_results - fetched),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            xml_text = self._get_with_retry(ARXIV_API_BASE, params)
            papers = list(self._parse_feed(xml_text))
            if not papers:
                break
            for p in papers:
                self._save_metadata(p)
                yield p
            fetched += len(papers)
            if len(papers) < batch_size:
                break
            time.sleep(self.rate_limit_s)

    def download_pdf(self, paper: ArxivPaper) -> Path | None:
        """Download PDF to raw_dir/<arxiv_id>.pdf; skip if already present."""
        dest = self.raw_dir / f"{paper.arxiv_id.replace('/', '_')}.pdf"
        if dest.exists():
            paper.pdf_path = str(dest)
            return dest
        try:
            url = f"{ARXIV_PDF_BASE}/{paper.arxiv_id}"
            resp = self._client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            paper.pdf_path = str(dest)
            self._save_metadata(paper)
            logger.info(f"Downloaded {paper.arxiv_id} → {dest}")
            time.sleep(self.rate_limit_s)
            return dest
        except Exception as exc:
            logger.warning(f"PDF download failed for {paper.arxiv_id}: {exc}")
            return None

    def load_cached(self) -> list[ArxivPaper]:
        """Return all previously crawled papers from JSON sidecar files."""
        papers = []
        for jf in sorted(self.raw_dir.glob("*.json")):
            try:
                data = json.loads(jf.read_text())
                papers.append(ArxivPaper(**data))
            except Exception as exc:
                logger.warning(f"Could not load {jf}: {exc}")
        return papers

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_query(
        categories: list[str],
        date_from: str | None,
        query_terms: list[str] | None = None,
    ) -> str:
        clauses = []

        cat_filter = " OR ".join(f"cat:{c}" for c in categories)
        clauses.append(f"({cat_filter})")

        if query_terms:
            # Match any term in title OR abstract. Multi-word phrases are quoted
            # so the arXiv API treats them as exact phrases.
            term_clauses = []
            for term in query_terms:
                t = f'"{term}"' if " " in term else term
                term_clauses.append(f"abs:{t}")
                term_clauses.append(f"ti:{t}")
            clauses.append("(" + " OR ".join(term_clauses) + ")")

        if date_from:
            # arXiv date filter uses submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM].
            # NOTE: an open-ended upper bound ("TO *") triggers HTTP 500 on the
            # arXiv API when combined with a multi-term OR block, so we always use
            # a concrete (far-future) upper bound instead.
            date_str = date_from.replace("-", "")
            clauses.append(f"submittedDate:[{date_str}0000 TO 20991231235959]")

        return " AND ".join(clauses)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _get_with_retry(self, url: str, params: dict) -> str:
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.text

    @staticmethod
    def _parse_feed(xml_text: str) -> Iterator[ArxivPaper]:
        root = ET.fromstring(xml_text)
        for entry in root.findall("atom:entry", NS):
            arxiv_id = (entry.findtext("atom:id", "", NS) or "").split("/abs/")[-1]
            if not arxiv_id:
                continue

            title = (entry.findtext("atom:title", "", NS) or "").strip().replace("\n", " ")
            abstract = (entry.findtext("atom:summary", "", NS) or "").strip()
            published = entry.findtext("atom:published", "", NS) or ""
            updated = entry.findtext("atom:updated", "", NS) or ""

            authors = [
                a.findtext("atom:name", "", NS) or ""
                for a in entry.findall("atom:author", NS)
            ]
            categories = [
                c.get("term", "")
                for c in entry.findall("atom:category", NS)
            ]

            pdf_url = ""
            for link in entry.findall("atom:link", NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")

            doi = entry.findtext("arxiv:doi", None, NS)
            journal_ref = entry.findtext("arxiv:journal_ref", None, NS)
            comments = entry.findtext("arxiv:comment", None, NS)

            yield ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=categories,
                published=published[:10],
                updated=updated[:10],
                pdf_url=pdf_url,
                doi=doi,
                journal_ref=journal_ref,
                comments=comments,
            )

    def _save_metadata(self, paper: ArxivPaper) -> None:
        dest = self.raw_dir / f"{paper.arxiv_id.replace('/', '_')}.json"
        dest.write_text(json.dumps(asdict(paper), indent=2))
