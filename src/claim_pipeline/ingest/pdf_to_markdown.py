from pathlib import Path


def convert_pdf_to_markdown(pdf_path: Path, dest_dir: Path) -> str:
    """Extracts plain text from a PDF via PyMuPDF, saves alongside the
    source, and returns the text.

    Swapped in for marker-pdf/surya: surya-ocr's legacy transformers APIs
    (transformers.onnx, transformers.pytorch_utils.find_pruneable_heads_and_indices,
    ...) keep breaking against transformers>=5.5.3, which vllm requires --
    no single transformers version satisfies both, so patching each newly
    broken import one at a time doesn't converge. Plain text extraction has
    no ML-model/transformers dependency at all, so it can't hit this class
    of conflict. Trade-off: loses marker's structure-aware output (tables,
    figures, equations rendered as clean markdown) -- acceptable given this
    pipeline doesn't need high accuracy here; clean_markdown()'s
    regex-based cleanup still applies fine to plain text since none of its
    patterns require literal markdown syntax to be present, they just
    no-op if it's absent.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    text = "\n\n".join(page.get_text("text") for page in doc)
    doc.close()

    dest_dir.mkdir(parents=True, exist_ok=True)
    md_path = dest_dir / f"{pdf_path.stem}_extracted.md"
    md_path.write_text(text, encoding="utf-8")

    return text
