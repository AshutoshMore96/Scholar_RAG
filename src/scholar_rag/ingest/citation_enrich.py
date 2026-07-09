"""
Citation enrichment — calls the OpenAlex API to attach citation counts,
references, concept tags, publication year and venue to each ingested paper.

OpenAlex is free, requires no API key (just a polite `mailto` User-Agent), and
has generous rate limits, so it is the sole citation-graph source. The influence
prior used by the graph-aware reranker is derived from OpenAlex's
`cited_by_count` (see store/graph_store.py and retrieve/graph_rerank.py).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


OPENALEX_API = "https://api.openalex.org"

# OpenAlex asks that you identify yourself in the User-Agent for the fast,
# rate-limit-friendly "polite pool". Set OPENALEX_MAILTO to a real contact.
_MAILTO = os.getenv("OPENALEX_MAILTO", "scholar@example.com").strip() or "scholar@example.com"
_UA = f"ScholarRAG/0.1 (mailto:{_MAILTO})"


@dataclass
class CitationMeta:
    arxiv_id: str
    openalex_id: str | None = None
    cited_by_count: int = 0
    year: int | None = None
    venue: str | None = None
    concepts: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)   # ids of referenced works

    @classmethod
    def from_dict(cls, data: dict) -> "CitationMeta":
        """Build from a dict, ignoring unknown keys (tolerates old cache schemas)."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


class CitationEnricher:
    """
    Enriches ArxivPaper metadata with citation-graph signals from OpenAlex.

    Stores enriched JSON sidecars alongside the raw paper JSON so the
    information is available both for Qdrant payload indexing and for
    the citation-graph DuckDB table.
    """

    def __init__(self, raw_dir: Path, rate_limit_s: float = 1.0) -> None:
        self.raw_dir = raw_dir
        self.rate_limit_s = rate_limit_s
        self._client = httpx.Client(
            timeout=30.0, follow_redirects=True, headers={"User-Agent": _UA}
        )

    def enrich(self, arxiv_id: str) -> CitationMeta:
        meta = CitationMeta(arxiv_id=arxiv_id)
        cache_path = self.raw_dir / f"{arxiv_id.replace('/', '_')}_citation.json"
        if cache_path.exists():
            return CitationMeta.from_dict(json.loads(cache_path.read_text()))

        self._fill_from_openalex(meta)
        time.sleep(self.rate_limit_s)

        cache_path.write_text(json.dumps(asdict(meta), indent=2))
        return meta

    def enrich_batch(self, arxiv_ids: list[str]) -> list[CitationMeta]:
        results = []
        for i, aid in enumerate(arxiv_ids):
            logger.info(f"Enriching {i + 1}/{len(arxiv_ids)}: {aid}")
            results.append(self.enrich(aid))
        return results

    # ------------------------------------------------------------------ #
    # OpenAlex                                                             #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    def _fill_from_openalex(self, meta: CitationMeta) -> None:
        try:
            work = self._lookup_work(meta.arxiv_id)
            if not work:
                return
            meta.openalex_id = work.get("id")
            meta.cited_by_count = work.get("cited_by_count", 0)
            meta.year = work.get("publication_year")
            primary = work.get("primary_location") or {}
            source = primary.get("source") or {}
            meta.venue = source.get("display_name")
            meta.concepts = [c["display_name"] for c in work.get("concepts", [])[:8]]
            for ref in (work.get("referenced_works") or []):
                ref_id = ref.split("/")[-1] if isinstance(ref, str) else ""
                if ref_id:
                    meta.references.append(ref_id)
        except Exception as exc:
            logger.warning(f"OpenAlex lookup failed for {meta.arxiv_id}: {exc}")

    def _lookup_work(self, arxiv_id: str) -> dict | None:
        """
        Resolve an arXiv id to an OpenAlex work.

        Primary path: the arXiv DOI (10.48550/arXiv.<id>), which OpenAlex indexes
        for arXiv preprints — the reliable key for freshly-crawled papers.
        Fallback path: filter on the arXiv landing-page URL, for records that
        still retain it (some do, some are merged under a published version).

        A transient 5xx raises so the tenacity retry on the caller kicks in; a
        genuine 404 (paper simply not indexed yet) returns None quietly.
        """
        # Strip any version suffix (e.g. 2405.12345v2 → 2405.12345); OpenAlex
        # indexes the base id.
        base_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id

        # 1. arXiv DOI lookup (DOIs are lowercased in OpenAlex).
        resp = self._client.get(
            f"{OPENALEX_API}/works/https://doi.org/10.48550/arxiv.{base_id}"
        )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code not in (404,):
            resp.raise_for_status()

        # 2. Landing-page-URL filter fallback.
        landing = f"https://arxiv.org/abs/{base_id}"
        resp2 = self._client.get(
            f"{OPENALEX_API}/works",
            params={"filter": f"locations.landing_page_url:{landing}", "per-page": 1},
        )
        if resp2.status_code == 200:
            results = resp2.json().get("results", [])
            if results:
                return results[0]
        elif resp2.status_code not in (404,):
            resp2.raise_for_status()

        return None
