GRAPH_GENERATION_TEMPLATE = """You are analyzing a set of atomic propositions extracted from a scientific paper. Determine the logical dependency structure: for each proposition, which OTHER propositions (among those given) must be known/established to derive or verify it?

Document summary (background context only -- covers the paper's main themes, use it to understand
*why* propositions relate to each other, e.g. one being part of the experimental design that made
another's finding possible):
{doc_summary}

Each proposition below is followed by the paper's own sentences it was derived from -- this is your
ONLY source of truth for whether a dependency is real. Do not use outside/parametric knowledge about
this topic, and do not infer a relationship that these sentences don't actually state or imply.

Propositions (with source sentences):
{proposition_block}

For every proposition, decide which other propositions (if any) are genuine logical prerequisites --
i.e. a reader could not verify or derive this proposition's truth without first knowing the
prerequisite. A dependency must be traceable to specific wording in the source sentences shown (e.g.
one proposition's sentences explicitly name, cause, or resolve something the other proposition
states) -- not merely that two propositions cover a similar theme, use similar words, or appear in
the same section.

Return ONLY valid JSON in this exact format, no other text -- do NOT restate the propositions
themselves, only the dependency edges between their ids:
{{
  "Edges": [
    {{"node": "P1", "inputs": ["P2", "P3"], "outputs": ["P4"], "evidence": ["S12", "S13"]}}
  ]
}}

Rules:
- Use only the proposition ids given above (e.g. "P1", "P2", ...) -- every id already exists, you do not need to declare or describe it.
- "inputs" = ids of propositions required to derive/verify this node. "outputs" = ids of propositions this node helps derive/verify.
- "evidence" = the sentence ids (from the source sentences shown above, e.g. "S12") that justify this specific edge -- the actual textual basis for the dependency, not just sentences near the topic. Required for every edge; if you cannot name specific sentence ids that justify it, do not include the edge at all.
- Do not include a node in its own "inputs" or "outputs".
- Omit "Edges" entries entirely for a node with no dependencies in either direction (do not emit empty-array edge entries).
- Do not include any edge not required for a genuine logical dependency. Two propositions being about the same topic, mentioned in the same section, or listed together in the paper is NOT by itself a dependency -- there must be an actual derivation/verification relationship (e.g. an experimental design choice that is a precondition for a finding being measurable at all), not mere thematic proximity.
- Purpose/goal phrasing ("X was done to achieve Y", "X in order to Y", "X to better Y") does NOT by itself tell you which of X or Y is the prerequisite for verifying the other -- the same sentence can be read either as X being evidence that the goal Y was pursued, or as Y being the context needed to understand why choice X was made, without settling which one you would need to check first. If purpose/goal phrasing is the ONLY signal connecting two propositions and the source sentences don't independently make clear which one is prerequisite to the other, omit the edge rather than guessing a direction -- treat this the same as not being able to cite specific evidence at all.
"""

CYCLE_REPAIR_TEMPLATE = """The following directed edges in a proposition dependency graph form a logical
cycle, which is invalid -- a proposition cannot (even transitively) require itself as a
prerequisite. Exactly one edge in this cycle is wrong, likely because the model picked an
incorrect direction for it, or the edge is spurious.

Cycle: {cycle_path}

Propositions and edges involved, each with the source evidence sentences that were cited for it:
{cycle_block}

Decide which ONE edge should be removed to break the cycle, based on which direction the cited
evidence actually supports. If more than one edge looks equally unsupported, prefer removing the
one with the weakest/fewest cited evidence sentences.

Return ONLY valid JSON in this exact format, no other text:
{{"remove_edge": {{"from": "P2", "to": "P3"}}, "reason": "one sentence justification"}}

The edge you return MUST be exactly one of the edges listed above (matching "from" and "to" ids
exactly)."""

EDGE_ACCURACY_TEMPLATE = """You are checking one edge in a proposition dependency graph. This requires
answering two SEPARATE questions -- evidence being present and topically relevant is not the same
as evidence establishing a genuine dependency in the specific claimed direction.

Proposition {from_id}: {from_description}
Proposition {to_id}: {to_description}

Claimed edge: {from_id} -> {to_id} (meaning: {to_id} depends on / requires {from_id} to derive or
verify it).

Cited evidence sentences:
{evidence_block}

Answer:
1. Does this evidence genuinely establish SOME dependency between these two propositions -- not
   just that both are mentioned in the same passage or share a topic? Accept causal or motivational
   reference language as sufficient (e.g. a pronoun or phrase in one sentence that explicitly points
   back to the other proposition's content -- "this observation," "as a result," "which suggests,"
   "motivated by" -- even if it falls short of strict logical necessity). Do NOT require that
   {to_id} would be false or undeterminable without {from_id} first being known -- that bar is
   stricter than what "genuine dependency" means here and will wrongly reject real reasoning-chain
   links the paper itself draws (e.g. one finding being offered as the stated motivation for
   measuring or reporting another). Only answer NO when the evidence presents the two propositions
   as separate, disjoint items with no textual link between them at all (e.g. a "some X... others
   Y..." list of distinct categories where neither is said to explain, cause, or motivate the
   other).
2. Does the evidence specifically support the claimed direction {from_id} -> {to_id}, or would the
   same evidence equally (or better) support the reverse direction {to_id} -> {from_id}? Look for
   explicit causal/derivational language (e.g. "X caused Y", "as a result of X", "Y required X",
   "X was measured using Y") rather than mere co-occurrence -- evidence that reads the same both
   ways does NOT support either direction over the other.

Return ONLY valid JSON in this exact format, no other text:
{{"score": 0, "direction_supported": true, "reasoning": "one or two sentences, explicitly addressing direction"}}

Scoring: 10 = evidence fully and unambiguously supports this specific direction. Around 5 = evidence
supports some dependency between these two propositions but does not clearly establish direction
over the reverse. 0 = evidence does not support any dependency between these two propositions at
all. Set "direction_supported" to true only if you would give this edge a score of 7 or higher."""

STRUCTURAL_FIDELITY_TEMPLATE = """You are assessing whether the overall shape of a proposition
dependency graph plausibly reflects how the paper's findings and arguments actually relate to each
other -- as opposed to looking arbitrary, or missing high-level connections the summary below
implies should exist.

IMPORTANT -- this graph's edges encode VERIFICATION/DERIVATION order, not physical or temporal
causal order: an edge A -> B means "a reader would need to know A to verify or derive B," NOT "A
physically/temporally causes B." These frequently point the same way (an experimental design choice
precedes the finding it enables, both in time and in what you'd need to know first), but they can
diverge for a proposition that itself states a specific causal attribution about an already-observed
event -- e.g. if proposition X is "some general problem occurred" and proposition Y is "cause C was
the specific reason X occurred," the standard here is X -> Y (you need to know the problem happened
before "C caused it" is even a claim that can be verified), even though C physically preceded X in
the world. Do NOT flag an edge as a "reversal" or structurally wrong solely because it points from a
general/observed fact to a specific cause attributed to it -- that is the correct direction under
this graph's own verification-order definition, not a flaw. Only flag genuine structural problems:
edges that don't follow from the summary at all, or gaps where the summary implies a connection that
is simply missing.

Document summary (background context on the paper's main themes and argument -- note this
necessarily compresses away fine-grained detail, so judge only what's visible at this level of
description, not anything more specific):
{doc_summary}

Graph nodes:
{node_block}

Graph edges (from -> to):
{edge_block}

A separate, edge-by-edge evidence check has already run and found the following edges do NOT hold up
-- their cited evidence does not actually support a dependency in the stated direction (co-occurrence
was mistaken for causation). Do not describe these specific edges as coherent, well-supported, or
evidence of good structure; treat them as known gaps/flaws when judging overall shape, the same way
you would treat a missing connection:
{rejected_edges_block}

Only critique or praise an edge that literally appears in the "Graph edges" list above or the
rejected-edges list above. Do NOT extend a critique of a rejected edge to a different edge just
because it looks structurally similar or shares a node -- e.g. if P16->P20 is listed as rejected but
P15->P20 is not in either list, say nothing about P15->P20; it was not evaluated, and claiming it
has "the same flaw" as a sibling edge states something as a finding that was never actually checked.

Return ONLY valid JSON in this exact format, no other text:
{{"score": 0, "reasoning": "..."}}

Scoring: 10 = structure is fully plausible and coherent given the summary, no obvious gaps or
arbitrary-looking connections, and no edges from the rejected list above. 5 = some plausible
structure but clear gaps, a few connections that don't obviously follow from the summary, or a
handful of rejected edges load-bearing to the overall shape. 0 = structure looks arbitrary,
contradicts the summary's account of how the paper's argument proceeds, or leans heavily on edges
already known to lack evidentiary support. Your reasoning must not cite a rejected edge as an example
of the graph's coherence."""
