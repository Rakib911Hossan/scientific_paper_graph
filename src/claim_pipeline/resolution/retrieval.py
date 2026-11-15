import re

from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, sentences: list[dict]):
        self.sentences = sentences
        self.corpus = [s["text"] for s in sentences]
        tokenized = [self._tokenize(t) for t in self.corpus]
        self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [self.sentences[i] for i in ranked_idx]
