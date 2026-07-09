"""
Streamlit UI for ScholarRAG.

Provides:
  - Research question input with optional filters (year, citations)
  - Cited literature review output with expandable source passages
  - Citation cards with paper metadata and arXiv links
  - Context quality score and pipeline debug info
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("SCHOLAR_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ScholarRAG — Academic Literature Search",
    page_icon="📚",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────── #
with st.sidebar:
    st.title("ScholarRAG")
    st.caption("Academic Knowledge Base • arXiv RAG Pipeline")
    st.divider()
    st.subheader("Retrieval filters")
    year_from = st.number_input("Year from", min_value=2000, max_value=2026, value=2020, step=1)
    year_to = st.number_input("Year to", min_value=2000, max_value=2026, value=2026, step=1)
    min_cit = st.number_input("Min. citations", min_value=0, value=0, step=5)
    top_k = st.slider("Passages to return", min_value=3, max_value=20, value=8)
    st.divider()
    st.caption("Powered by BGE-M3 + Ollama + Qdrant")

# ── Main ─────────────────────────────────────────────────────────────── #
st.title("📚 ScholarRAG — Literature Review Generator")
st.caption(
    "Ask a research question. ScholarRAG retrieves relevant arXiv papers, "
    "reranks them by relevance and citation influence, and generates a "
    "traceable literature review with inline citations."
)

query = st.text_area(
    "Research question",
    placeholder=(
        "e.g. What are the trade-offs between late-interaction and bi-encoder "
        "retrieval for long documents?"
    ),
    height=100,
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_btn = st.button("Ask", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=False):
        st.rerun()

if ask_btn and query.strip():
    with st.spinner("Retrieving and generating cited literature review…"):
        try:
            resp = httpx.post(
                f"{API_URL}/ask",
                json={
                    "query": query.strip(),
                    "year_from": int(year_from) if year_from else None,
                    "year_to": int(year_to) if year_to else None,
                    "min_citations": int(min_cit) if min_cit else None,
                    "top_k": top_k,
                },
                timeout=420.0,   # CPU inference (esp. CRAG retry / cold model load) is slow
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            st.error(f"API error: {exc}")
            st.stop()

    # ── Literature review ─────────────────────────────────────── #
    st.divider()
    if data["abstained"]:
        st.warning(data["review"])
    else:
        st.subheader("Literature Review")
        st.markdown(data["review"])

        quality = data["context_quality"]
        q_color = "green" if quality > 0.7 else "orange" if quality > 0.4 else "red"
        st.caption(
            f"Context quality: :{q_color}[{quality:.2f}] • "
            f"Latency: {data['latency_ms']} ms • "
            f"Citations used: {len(data['citations'])}"
        )

    # ── Citation cards ────────────────────────────────────────── #
    if data["citations"]:
        st.divider()
        st.subheader("Sources")
        for cit in data["citations"]:
            with st.expander(
                f"[{cit['paper_id']}] {cit['title'] or cit['paper_id']} "
                f"({cit['year'] or '?'}) — score: {cit['score']:.3f}"
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    if cit["venue"]:
                        st.caption(f"Venue: {cit['venue']}")
                    st.caption(f"Year: {cit['year']}")
                with col_b:
                    arxiv_id = cit["paper_id"].replace("_", "/")
                    st.link_button(
                        "Open on arXiv",
                        url=f"https://arxiv.org/abs/{arxiv_id}",
                    )

elif ask_btn:
    st.warning("Please enter a research question.")
