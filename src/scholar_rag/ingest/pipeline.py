"""
Ingestion orchestrator — ties together the crawler, citation enricher,
and hands off to the transformation pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from scholar_rag.config import load_config
from scholar_rag.ingest.arxiv_crawler import ArxivCrawler
from scholar_rag.ingest.citation_enrich import CitationEnricher


def run_ingestion(cfg: dict[str, Any] | None = None) -> list[dict]:
    """
    Full ingestion run.  Returns a list of dicts with keys:
      paper, citation_meta
    ready for the transformation stage.
    """
    if cfg is None:
        cfg = load_config()

    ing = cfg["ingestion"]
    raw_dir = Path(cfg.get("DATA_RAW_DIR", "data/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    crawler = ArxivCrawler(raw_dir=raw_dir, rate_limit_s=ing["rate_limit_delay_s"])
    enricher = CitationEnricher(raw_dir=raw_dir)

    query_terms = ing.get("query_terms")
    if query_terms:
        logger.info(f"Topic filter active — terms: {query_terms}")
    logger.info(f"Fetching up to {ing['max_results_per_query']} papers per category: {ing['categories']}")
    all_papers = []
    for category in ing["categories"]:
        for paper in tqdm(
            crawler.fetch(
                categories=[category],
                max_results=ing["max_results_per_query"],
                date_from=ing.get("date_from"),
                query_terms=query_terms,
            ),
            desc=f"Crawling {category}",
            unit="paper",
        ):
            all_papers.append(paper)
            if ing["download_pdfs"]:
                crawler.download_pdf(paper)

    logger.info(f"Crawled {len(all_papers)} papers.  Enriching citations…")
    records = []
    for paper in tqdm(all_papers, desc="Citation enrichment", unit="paper"):
        meta = enricher.enrich(paper.arxiv_id)
        records.append({"paper": paper, "citation_meta": meta})

    logger.success(f"Ingestion complete: {len(records)} records ready for transformation.")
    return records
