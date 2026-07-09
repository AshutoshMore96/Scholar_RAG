"""
Nougat-based PDF parser.  Nougat converts academic PDFs to Markdown,
preserving LaTeX equations, tables, and figure captions.

Nougat must be installed separately:
    pip install nougat-ocr
    # OR run it as an HTTP microservice (recommended for batch):
    nougat_api  (see: https://github.com/facebookresearch/nougat)
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from loguru import logger


class NougatParser:
    """
    Wraps the Nougat CLI.  Calls `nougat <pdf>` and returns the resulting
    Markdown text.  Falls back to None on failure so the orchestrator can
    route to PyMuPDF / GROBID.
    """

    def __init__(self, batch_size: int = 4) -> None:
        self.batch_size = batch_size
        self._available = self._check_nougat()

    @property
    def available(self) -> bool:
        return self._available

    def parse(self, pdf_path: Path) -> str | None:
        if not self._available:
            return None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(
                    ["nougat", str(pdf_path), "--out", tmpdir, "--no-skipping"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    logger.warning(f"Nougat non-zero exit for {pdf_path.name}: {result.stderr[:200]}")
                    return None
                mmd_files = list(Path(tmpdir).glob("*.mmd"))
                if not mmd_files:
                    return None
                return mmd_files[0].read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Nougat parse failed for {pdf_path.name}: {exc}")
            return None

    def parse_batch(self, pdf_paths: list[Path]) -> dict[Path, str | None]:
        return {p: self.parse(p) for p in pdf_paths}

    @staticmethod
    def _check_nougat() -> bool:
        try:
            result = subprocess.run(["nougat", "--version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except FileNotFoundError:
            logger.info("Nougat not found — will use fallback parsers.")
            return False
