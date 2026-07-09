"""Prompt templates for the generation stage."""

LITERATURE_REVIEW_SYSTEM = """\
You are an expert research scientist writing a literature review.
You have been given a set of retrieved passages from academic papers.
Your task is to synthesise these passages into a concise, accurate, and
well-structured literature review that directly answers the research question.

STRICT RULES:
1. Every factual claim MUST be followed by a citation in the format [paper_id].
2. Do NOT introduce any information not present in the provided passages.
3. If the passages do not contain sufficient information to answer the question,
   respond with: "INSUFFICIENT EVIDENCE: <brief explanation>"
4. Use precise academic language.
5. Structure your response as:
   - A 1-2 sentence direct answer to the question
   - A synthesis of key findings with citations
   - A brief note on research gaps or open questions (if evident from the passages)
"""

LITERATURE_REVIEW_USER = """\
Research question: {query}

Retrieved passages (each begins with its citation id in square brackets):
{context}

Write a literature review (150–300 words) answering the research question.

CITATION FORMAT — follow exactly:
- End EVERY sentence with the bracketed id of the passage it draws from, e.g.
  "Dense retrieval encodes queries and documents into a shared space [2401.01234]."
- Copy the ids verbatim from the passages above. Use only those ids.
- A sentence may cite more than one id: "...improves recall [2401.01234][2402.05678]."

Example of the required style:
"Retrieval-augmented generation grounds outputs in retrieved passages [2312.10997].
Hybrid retrieval combines dense and sparse signals to improve recall [2401.01234]."

Now write the review, citing after every sentence:
"""

SELF_RAG_REFLECTION_PROMPT = """\
Review the following generated literature review and assess each factual claim.
For each claim, verify it is directly supported by at least one cited passage.

Generated review:
{review}

Retrieved passages (for reference):
{context}

For each unsupported claim (not backed by a passage), mark it with [UNSUPPORTED].
Output the corrected review, dropping any [UNSUPPORTED] claims.
Output ONLY the corrected review text, no explanation.

Corrected review:"""

ABSTAIN_THRESHOLD_PROMPT = """\
On a scale of 0.0 to 1.0, how well do the following passages support a
comprehensive answer to the research question?

Question: {query}

Passages: {snippets}

Output ONLY a floating-point number. Score:"""
