"""
Citation graph store backed by DuckDB.

Schema
------
papers(arxiv_id, year, cited_by_count, influential_citation_count, concepts)
cites(src_id, dst_id)   — directed edge: src paper cites dst paper

The graph-reranker reads from this table to compute influence priors.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import duckdb
from loguru import logger


_DEFAULT_DB = Path("data/citations.duckdb")


class CitationGraphStore:
    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(db_path))
        self._init_schema()

    # ------------------------------------------------------------------ #

    def _init_schema(self) -> None:
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                arxiv_id                  VARCHAR PRIMARY KEY,
                year                      INTEGER,
                cited_by_count            INTEGER DEFAULT 0,
                influential_citation_count INTEGER DEFAULT 0,
                venue                     VARCHAR,
                concepts                  VARCHAR[]
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS cites (
                src_id VARCHAR,
                dst_id VARCHAR,
                PRIMARY KEY (src_id, dst_id)
            )
        """)
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_cites_dst ON cites(dst_id)")

    def upsert_paper(self, arxiv_id: str, meta: dict[str, Any]) -> None:
        self.con.execute("""
            INSERT OR REPLACE INTO papers
              (arxiv_id, year, cited_by_count, influential_citation_count, venue, concepts)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            arxiv_id,
            meta.get("year"),
            meta.get("cited_by_count", 0),
            meta.get("influential_citation_count", 0),
            meta.get("venue"),
            meta.get("concepts", []),
        ])

    def upsert_edges(self, src_id: str, dst_ids: list[str]) -> None:
        rows = [(src_id, dst) for dst in dst_ids if dst]
        if rows:
            self.con.executemany(
                "INSERT OR IGNORE INTO cites (src_id, dst_id) VALUES (?, ?)", rows
            )

    def get_influence_prior(self, arxiv_id: str, current_year: int = 2026) -> dict[str, float]:
        """
        Returns:
          log_influence  = log(1 + cited_by_count)   [from OpenAlex]
          recency        = 1 / (1 + years_since_publication)
        """
        row = self.con.execute(
            "SELECT year, cited_by_count FROM papers WHERE arxiv_id = ?",
            [arxiv_id],
        ).fetchone()
        if not row:
            return {"log_influence": 0.0, "recency": 0.5}
        year, cited = row
        years_old = max(0, current_year - (year or current_year))
        return {
            "log_influence": math.log1p(cited or 0),
            "recency": 1.0 / (1.0 + years_old),
        }

    def bulk_load(self, citation_metas: list[dict[str, Any]]) -> None:
        for meta in citation_metas:
            self.upsert_paper(meta["arxiv_id"], meta)
            self.upsert_edges(meta["arxiv_id"], meta.get("references", []))
        logger.success(f"Loaded {len(citation_metas)} papers into citation graph.")
