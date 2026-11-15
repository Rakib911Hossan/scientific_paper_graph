import re


def clean_markdown(md: str) -> str:
    text = md

    # Cut everything before "Abstract" (drops title/author-block noise).
    # Some PDFs render the heading as bold ("# **Abstract**") rather than
    # plain text, so optional surrounding "**" markers are tolerated too.
    abstract_match = re.search(r"^#*\s*\**\s*Abstract\s*\**\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    if abstract_match:
        text = text[abstract_match.start() :]

    # marker sometimes renders a citation's own parens as literal `\(`/`\)` escapes
    # split across two+ adjacent page-anchor links, e.g.
    # [\(Wadden et al.,](#page-11-0) [2020\)](#page-11-0) -- unescape those before the
    # anchor-stripping regex below runs, so it doesn't wrap already-parenthesized
    # citation text in a second, redundant layer of parens.
    text = text.replace("\\(", "(").replace("\\)", ")")

    # Remove citation anchors like [Liang et al., 2022](#page-9-1) -> (Liang et al., 2022).
    # Tracks paren depth across consecutive matches: once a citation's own opening
    # paren has been seen (per the unescape above) and not yet closed, later groups
    # in the same citation/citation-chain are emitted as-is instead of each being
    # wrapped in their own parens, which would double up
    # (e.g. avoids "((Press et al., 2023; Khot et al., 2023; Chen et al., 2022))").
    def _strip_page_anchor(match, _depth=[0]):
        inner = match.group(1)
        opens, closes = inner.count("("), inner.count(")")
        if _depth[0] > 0 or opens > closes:
            _depth[0] += opens - closes
            return inner
        if opens > 0 and opens == closes:
            # single group carrying its own already-balanced escaped parens,
            # e.g. [\(Akhtar et al.\)](#page-8-2) -> "(Akhtar et al.)" already
            return inner
        return f"({inner})"

    text = re.sub(r"\[([^\]]+)\]\(#page-\d+-?\d*\)", _strip_page_anchor, text)

    # Remove raw markdown links but keep the link text: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1", text)

    # Remove span/anchor HTML tags
    text = re.sub(r"<span[^>]*>.*?</span>", "", text, flags=re.DOTALL)
    text = re.sub(r"<span[^>]*/?>", "", text)

    # Remove sup/sub tags but keep inner text
    text = re.sub(r"</?su[pb]>", "", text)

    # Remove image references
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # --- SELECTIVE figure/table caption handling ---
    # Instead of deleting every "Figure N: ..." / "Table N: ..." line outright,
    # keep it if it contains a number (likely real evidence), strip it if not
    # (likely just a label or pure description with no data).
    def _handle_caption(match):
        line = match.group(0)
        has_number = bool(re.search(r"-?\d+\.?\d*%?", line))
        return line if has_number else ""

    text = re.sub(r"^(Figure|Table)\s+\d+:.*$", _handle_caption, text, flags=re.MULTILINE)

    # Remove stray #page- refs
    text = re.sub(r"#page-\d+-?\d*", "", text)

    # --- SELECTIVE equation handling ---
    # Keep equations that contain actual numeric constants (likely load-bearing,
    # e.g. weighting formulas like 1/3, 2/3), drop pure symbolic/variable-only ones.
    def _handle_equation(match):
        eq = match.group(0)
        has_digit = bool(re.search(r"\d", eq))
        if has_digit:
            cleaned_eq = eq.strip("$")
            cleaned_eq = cleaned_eq.replace("\\times", "×")
            cleaned_eq = cleaned_eq.replace("\\left", "").replace("\\right", "")
            # Convert \frac{a}{b} -> a/b BEFORE stripping braces, or numerator/denominator run together
            cleaned_eq = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", cleaned_eq)
            cleaned_eq = re.sub(r"[{}\\]", "", cleaned_eq)
            return f" [Formula: {cleaned_eq.strip()}] "
        return ""

    text = re.sub(r"\$\$.*?\$\$", _handle_equation, text, flags=re.DOTALL)
    text = re.sub(r"\$[^\$\n]+\$", _handle_equation, text)

    # Remove markdown tables (pipe-delimited rows and separator rows)
    # NOTE: table content itself (e.g. package counts) is still dropped here,
    # since tables need structured handling, not sentence-level extraction.
    # If you want table stats preserved too, they should be extracted separately
    # as structured data, not shoehorned into sentence tokenization.
    text = re.sub(r"^\|.*\|\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s\-\|]+$", "", text, flags=re.MULTILINE)

    # Remove markdown header markup but keep text
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace per line
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def remove_references_section(text: str) -> str:
    match = re.search(r"^\s*References\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    if match:
        text = text[: match.start()]
    return text.strip()
