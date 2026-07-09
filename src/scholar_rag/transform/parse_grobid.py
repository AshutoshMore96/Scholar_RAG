"""
GROBID-based PDF parser.  GROBID extracts structured TEI XML from PDFs;
we convert it to Markdown-like plain text with section headers preserved.
GROBID runs as a Docker service on port 8070.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from loguru import logger


_GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")


class GrobidParser:
    def __init__(self, grobid_url: str = _GROBID_URL) -> None:
        self.url = grobid_url.rstrip("/")
        self._client = httpx.Client(timeout=120.0)

    @property
    def available(self) -> bool:
        try:
            resp = self._client.get(f"{self.url}/api/isalive", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def parse(self, pdf_path: Path) -> str | None:
        try:
            with pdf_path.open("rb") as fh:
                resp = self._client.post(
                    f"{self.url}/api/processFulltextDocument",
                    files={"input": (pdf_path.name, fh, "application/pdf")},
                    data={"consolidateHeader": "1", "consolidateCitations": "0", "includeRawCitations": "1"},
                )
            resp.raise_for_status()
            return self._tei_to_markdown(resp.text)
        except Exception as exc:
            logger.warning(f"GROBID parse failed for {pdf_path.name}: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # TEI → Markdown conversion                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tei_to_markdown(tei_xml: str) -> str:
        """Convert GROBID TEI XML to a simple Markdown representation."""
        from xml.etree import ElementTree as ET
        import re

        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError:
            return tei_xml  # return raw on parse failure

        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        lines: list[str] = []

        # Title
        title_el = root.find(".//tei:titleStmt/tei:title", ns)
        if title_el is not None and title_el.text:
            lines.append(f"# {title_el.text.strip()}\n")

        # Abstract
        abstract_el = root.find(".//tei:abstract", ns)
        if abstract_el is not None:
            abstract_text = "".join(abstract_el.itertext()).strip()
            lines.append(f"## Abstract\n\n{abstract_text}\n")

        # Body sections
        body = root.find(".//tei:body", ns)
        if body is not None:
            for div in body.findall(".//tei:div", ns):
                head = div.find("tei:head", ns)
                if head is not None and head.text:
                    lines.append(f"\n## {head.text.strip()}\n")
                for p in div.findall("tei:p", ns):
                    text = "".join(p.itertext()).strip()
                    if text:
                        # Preserve formula references as [FORMULA]
                        text = re.sub(r"\s+", " ", text)
                        lines.append(f"{text}\n")

        return "\n".join(lines)
