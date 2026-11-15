EVIDENCE_QUESTIONS_TEMPLATE = """You are preparing supporting evidence for a claim extracted from a specific research paper.

Document summary (background context only -- covers the paper's main themes, may not mention
this specific claim's topic at all):
{doc_summary}

Sentences from the paper most likely related to this specific claim:
{candidate_block}

Claim: {claim}

Generate up to {n} distinct questions whose answers can be directly supported by the candidate
sentences above. Only fall back to the document summary if the candidate sentences genuinely lack
the context needed -- do not reach for the document summary just because it is easier to answer
from; the candidate sentences are the primary source and should be exhausted first.

Each question must target a DIFFERENT piece of evidence -- do not ask two questions that would be
answered by the same sentence, and do not pad the list with a redundant or overly generic question
just to reach {n}. Use your judgment on how many distinct pieces of evidence this specific claim
genuinely warrants -- a narrow, single-fact claim may only need 1 question; a claim combining
several distinct facts (e.g. a cost claim citing both a dollar figure and a time figure) may
warrant more, up to {n}. Always include at least 1.

PREFER STRONGER EVIDENCE: when multiple valid questions are possible, prefer ones whose answers
draw on stronger evidence, in this order of preference:
  1. Experimental results
  2. Quantitative measurements
  3. Ablation findings
  4. Method descriptions
  5. Implementation details
  6. Background information
Do not settle for a weak background-level fact if the candidate sentences contain a stronger
result, measurement, or ablation finding that supports the same claim.

BEFORE FINALIZING EACH QUESTION, check it against BOTH of the following -- discard or rewrite the
question if either fails:
  (a) Not a restatement: would the answer supply a NEW concrete detail (a number, name,
      mechanism, or specific outcome) not already implied by the claim's wording? If the answer
      would just echo the claim back in different words, discard it or sharpen the question.
  (b) Actually relevant: does the answer directly support THIS claim specifically, not just
      relate to the same general topic? A concrete, correctly-answered fact that doesn't bear on
      the claim is still a bad question -- concreteness alone is not enough.

Examples:
  Claim: "The proposed method reduces memory usage while maintaining accuracy."
  BAD (fails a -- restatement): Q: "What did the paper find about memory and accuracy?"
                                 A: "The method reduces memory usage while maintaining accuracy." -> just echoes the claim.
  BAD (fails b -- irrelevant):  Q: "What programming language was the implementation written in?"
                                 A: "Python." -> concrete, but doesn't support a claim about memory/accuracy.
  GOOD (passes both):           Q: "What memory reduction and accuracy change were measured relative to the baseline?"
                                 A: "Peak memory dropped from 14.2GB to 9.1GB with less than 0.3% accuracy loss." -> concrete
                                 result that directly supports the claim.

IMPORTANT: Only ask about evidence you have reason to believe this paper actually contains, based
on the sentences and document summary above. Do not ask about specific results, model names, or
numbers from other well-known papers on this topic unless they explicitly appear above -- if
you're not sure this paper reports a specific number, ask a more general question instead of
guessing a plausible-sounding specific one. If the candidate sentences above don't actually
support the claim's specific topic, prefer a more general question over inventing specifics.

Return ONLY valid JSON in this exact format, no other text:
{{"questions": ["question 1", "question 2", ...]}}
"""

ATOMIC_CLAIMS_TEMPLATE = """You are converting a piece of evidence into standalone, checkable claims.

Original claim (context only, to help resolve pronouns/vague references -- do not repeat this
claim verbatim in your output):
{claim}

Question: {question}
Answer: {answer}

Rewrite the answer above into one or more standalone, self-contained, declarative claim sentences
that state the fact(s) it contains. Each sentence must be understandable entirely on its own, with
no dangling pronouns ("they", "it", "this") or vague references ("the winning teams", "the
competition") left unresolved -- replace them with what they actually refer to, using the original
claim and question for context.

Rules:
- If the answer contains multiple distinct facts (e.g. a list of separate methods, filters, or
  steps), split them into separate claims -- one fact per claim, do not combine them.
- If the answer is a single fact, produce exactly one claim.
- When splitting a PAIRED list (e.g. "X and Y drop by A and B respectively", where each value maps
  to one specific item in a shared list), each resulting claim must carry ONLY the one item its own
  value actually belongs to -- do not keep the full shared list/label in every split claim, since
  that makes the value's specific referent impossible to recover from the claim text alone. For
  example, from "coverage and diversity drop by 3.9 and 3.2 respectively", produce "coverage drops
  by 3.9" and "diversity drops by 3.2" -- NOT two claims that both say "coverage and diversity" with
  only the number differing.
- Preserve all numbers, units, named entities, and technical terms exactly as written in the
  answer -- do not paraphrase or round them.
- Do not add any fact, number, or entity that isn't present in the answer or the original
  claim/question.
- Keep each claim as short and direct as possible while remaining a complete, standalone sentence.
- Phrase every claim as a DIRECT assertion of fact, not as a meta-statement about the paper's
  rhetoric -- write "X" or "X can do Y", not "the paper suggests/argues/recommends that X" or "the
  authors state that X". The claim should assert the underlying fact/finding/recommendation
  itself, not the act of the paper stating it. For example, prefer "Retrieval-augmented generation
  can complement fine-tuning in real-world AI systems" over "The paper suggests combining
  retrieval-augmented generation with fine-tuning."

Return ONLY valid JSON in this exact format, no other text:
{{"claims": ["claim 1", "claim 2", ...]}}
"""

FINAL_FILTER_TEMPLATE = """You are filtering a list of claims extracted from a research paper, keeping
only the ones that state an objective, checkable fact worth independently verifying.

Claims (numbered):
{numbered_claims}

For each claim, decide whether to KEEP or DROP it:
- KEEP: the claim states a concrete, checkable fact -- a measurement, a specific number, a
  described method or step, a named entity/tool that was used, or a general finding the paper's
  evidence directly supports.
- DROP: the claim is an opinion, interpretation, or subjective judgment stated as commentary
  rather than a checkable fact (e.g. "this emphasizes the importance of X", "this is unexpected",
  "the authors believe"); a hedge about whether something is true in the WORLD ("may indicate",
  "suggests but is not certain", "the authors expected to see X" as a prediction that may or may
  not have held); a personal-experience or narrative aside that is not itself a checkable claim;
  or a bare reference to a figure/table without stating the actual fact it shows (e.g. "Figure 7
  shows X" should be dropped in favor of the claim that states X directly, if one exists -- if
  this claim is the only place X's content is captured, keep it instead of dropping the fact
  entirely). Do NOT drop a claim just because it is phrased as "the paper suggests/argues/
  recommends X" -- that is a definite, checkable fact about what the paper's content says (keep
  it, ideally treating X itself as the checkable claim), not an epistemic hedge about the world.
- Do NOT drop a claim just because it is a qualitative/comparative descriptor with no attached
  number ("low inference cost", "high accuracy", "significantly faster") -- if the paper states
  this as one of its own descriptive facts (not the claim-writer's own editorializing about the
  paper), it is exactly as checkable as a claim with a specific figure attached: a reader can
  still verify whether the paper supports it. Only treat a qualitative claim as a subjective
  opinion/DROP if it is evaluative commentary ABOUT the paper or its findings (e.g. "this result
  is impressive"), not a plain description the paper itself makes about its subject.
- When a claim is borderline, prefer KEEP if it states any concrete, checkable content at all, and
  DROP only the parts of the list that are purely editorial/interpretive with no checkable content
  of their own.
- Do not rewrite or alter any claim's wording -- only decide keep or drop.

For EVERY claim, give a decision and a reason of no more than 10 words citing the specific
KEEP/DROP criterion above that applies to THIS claim -- a reason that would read identically
across different claims (e.g. "concrete checkable fact" repeated verbatim) means you pattern-matched
instead of checking this claim individually, and must be made specific (name the actual fact,
number, or phrasing that drove the decision).

Return ONLY valid JSON in this exact format, no other text -- include every claim number exactly
once, in original order:
{{"decisions": [{{"index": 1, "decision": "keep", "reason": "..."}}, {{"index": 2, "decision": "drop", "reason": "..."}}, ...]}}
"""

DEDUP_TEMPLATE = """You are deduplicating a list of claims extracted from a research paper. These
claims were generated independently from different question/answer pairs, so the same underlying
fact can appear more than once, worded differently or at a different level of detail.

Claims (numbered):
{numbered_claims}

Find groups of claims that state the SAME underlying fact -- not just the same topic, but the same
specific fact, number, or finding, even if phrased with different words or different levels of
detail (e.g. "dropped to chance level" and "performed quite poorly, at about chance level" are the
same fact; "the model was fine-tuned" and "the model uses QLoRA for fine-tuning" are NOT the same
fact, since the second is more specific/different content, not a mere rewording).

For each group of 2 or more duplicate claims, decide which single claim to KEEP -- prefer the one
that states the fact most completely/specifically, as long as it doesn't add anything the others
lack support for. Do not merge or rewrite text; only choose which existing claim to keep.

For each group, also give a `reason` of no more than 12 words naming the SPECIFIC shared fact or
number the group's claims restate (e.g. "both cite the 4.8-point balanced-accuracy drop"), not a
generic statement like "these claims are duplicates" or "same topic" -- a reason that would read
identically across different groups means you have not actually compared their content and must
look again.

Claims not part of any duplicate group should not appear in your output at all.

Return ONLY valid JSON in this exact format, no other text:
{{"duplicate_groups": [{{"indices": [1, 24], "keep": 24, "reason": "..."}}, ...]}}
"""
