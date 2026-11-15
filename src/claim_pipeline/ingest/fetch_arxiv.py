from pathlib import Path

import requests


def fetch_arxiv_pdf(arxiv_id: str, dest_dir: Path) -> Path:
    """Downloads a paper PDF by arXiv id into dest_dir, returns the local path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = dest_dir / f"{arxiv_id}.pdf"

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    response = requests.get(pdf_url)
    response.raise_for_status()
    pdf_path.write_bytes(response.content)

    return pdf_path
