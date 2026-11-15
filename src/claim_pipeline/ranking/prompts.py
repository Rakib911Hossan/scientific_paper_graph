THESIS_GENERATION_TEMPLATE = """You are analyzing a research paper. Below is a short summary, followed by every sentence extracted from the paper's body text.

Document summary:
{doc_summary}

Your task: identify up to {max_claims} distinct, high-level findings or conclusions this paper actually supports. Each one should be a standalone thesis statement a reader could understand without the paper in front of them -- not a direct quote, but also not vaguer or broader than what the paper's own evidence justifies. Do not fabricate or overstate -- every claim must be traceable to specific content below.

Rules:
- Each claim must be about a DIFFERENT finding -- no two claims should restate the same underlying point in different words. If you can't find {max_claims} truly distinct findings, return fewer rather than padding with near-duplicates.
- Prefer claims that combine a general finding with why it matters or what caused it, when the paper supports that connection (e.g. "X happened because Y" rather than just "X happened"), but do not editorialize beyond what's stated or implied by the paper's own evidence.
- A finding can be empirical ("X was observed") OR prescriptive/recommendation-based ("the paper argues/recommends that X should be done") -- both count as findings as long as the paper actually states or argues for them. Only exclude claims that are pure procedural description with no finding, argument, or recommendation attached (e.g. "the competition had two tracks" on its own, with no conclusion drawn from it).

Sentences:
{sentence_block}

Return ONLY valid JSON in this exact format, no other text:
{{
  "claims": ["claim 1 text", "claim 2 text", ...]
}}
"""

THESIS_MERGE_TEMPLATE = """Below is a numbered list of candidate findings extracted independently from different sections of the same paper (chunked generation -- see ranking/chunking.py) -- some may restate the same underlying point in different words, since they were generated without visibility into each other.

Document summary:
{doc_summary}

Candidates:
{numbered_candidates}

Your task: select up to {max_claims} distinct, high-level findings from this list. Merge near-duplicates into a single best-phrased version (prefer the more complete/specific phrasing). No two selected claims should restate the same underlying point. Each selected claim should be a standalone thesis statement a reader could understand without the paper in front of them.

Return ONLY valid JSON in this exact format, no other text:
{{
  "claims": ["claim 1 text", "claim 2 text", ...]
}}
"""
