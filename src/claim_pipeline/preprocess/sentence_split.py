import re

import nltk


def _ensure_nltk_data():
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)


def is_section_header(sent: str) -> bool:
    # e.g. "1 The competition", "1.1 Format", "3.2 Software quality",
    # "2.2 Meta-Analysis", "3.4 Utility of Fine-Tuning"
    return bool(re.match(r"^\d+(\.\d+)*\s+[A-Za-z][A-Za-z\s\-]{0,60}$", sent))


def is_chart_junk(sent: str) -> bool:
    # e.g. "0.8 variation Set MWR"
    return bool(re.match(r"^\d+\.\d+\s+\w[\w\s]{0,20}$", sent))


def is_bare_cross_reference(sent: str) -> bool:
    # e.g. "A detailed analysis of this adaptation is presented in Section (2.2.)"
    # e.g. "See Section 3.2." / "Further details are in Table 4."
    return (
        bool(re.search(r"\b(is presented|is discussed|is shown|see)\b.{0,30}\bsection\b", sent, re.IGNORECASE))
        and len(sent) < 100
    )


def is_boilerplate(sent: str) -> bool:
    # Handle both ASCII * and Unicode ∗ (U+2217) used for footnote markers
    if re.match(r"^[\*∗]?\s*Corresponding author", sent, re.IGNORECASE):
        return True
    if re.match(r"^See\s+<https?://", sent, re.IGNORECASE):
        return True
    # Bare footnote-style URL references, e.g. "3 https://github.com/..."
    if re.match(r"^\d+\s+https?://", sent):
        return True
    return False


def split_into_sentences_with_ids(text: str) -> list[dict]:
    _ensure_nltk_data()
    from nltk.tokenize import sent_tokenize

    # Protect "et al." and common abbreviations from being treated as sentence enders
    protected = text.replace("et al.", "et al<PERIOD>")
    protected = re.sub(r"\b(Fig|Eq|Sec|No|vs|etc|approx)\.", r"\1<PERIOD>", protected)

    paragraphs = [p for p in protected.split("\n\n") if p.strip()]

    sentences = []
    sid = 1
    seen = set()  # dedupe exact repeats (e.g. repeated chart labels)
    for para in paragraphs:
        para = para.replace("\n", " ").strip()
        if not para:
            continue
        for sent in sent_tokenize(para):
            sent = sent.replace("<PERIOD>", ".").strip()

            if len(sent) < 15:
                continue
            if is_section_header(sent):
                continue
            if is_chart_junk(sent):
                continue
            if is_bare_cross_reference(sent):
                continue
            if is_boilerplate(sent):
                continue
            if sent in seen:
                continue

            seen.add(sent)
            sentences.append({"id": f"S{sid}", "text": sent})
            sid += 1

    return sentences
