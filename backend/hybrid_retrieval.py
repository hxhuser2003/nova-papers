"""R10 (inference-only half): BM25 + dense retrieval fused with Reciprocal
Rank Fusion, swapped in for QasperCorpus.search() without touching the
production corpus, agent_loop, or training path.

Reference: paperlens (github.com/EnesUguzz/paperlens) uses the same
BM25 + bge-m3 + RRF design for QASPER; their own ablation found adding a
reranker on top gave minimal additional gain, so this stays hybrid-only
for now (see TRAINING_NOTES.md's R10 entry) -- reranking would be a
follow-up experiment layered on top of this, not bundled with it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from backend.qasper_corpus import SEARCH_LIMIT, QasperCorpus

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> list[int]:
    """Each ranking is a list of item indices, best first. Standard RRF:
    score(d) = sum over rankers of 1/(k + rank(d)), rank starting at 1.
    k=60 is the de facto default used across most RRF implementations and
    is not itself tuned for QASPER -- noted as a parameter, not a fixed
    constant, in case it needs revisiting later."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda index: scores[index], reverse=True)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SentenceTransformerEmbedder:
    """Lazy-loaded real embedder. Only imported/constructed when actually
    used (CARC's inference venv, not this repo's test environment) --
    mirrors train_grpo.py's lazy trl/vllm imports."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        return self.model.encode(list(texts), convert_to_numpy=False)


class HybridQasperCorpus(QasperCorpus):
    """Same search()/read()/find_paper() surface as QasperCorpus; only
    search() ranks differently (BM25 + dense, fused with RRF) -- read(),
    find_paper(), .papers, .questions are inherited unchanged."""

    def __init__(self, directory: Path, embedder: Embedder, rrf_k: int = 60, search_limit: int = SEARCH_LIMIT):
        super().__init__(directory, search_limit=search_limit)
        self._embedder = embedder
        self._rrf_k = rrf_k
        self._dense_index: dict[str, list[list[float]]] = {}

    def _dense_vectors_for(self, paper_id: str) -> list[list[float]]:
        if paper_id not in self._dense_index:
            paragraphs = self.papers[paper_id]["paragraphs"]
            texts = [p["text"] for p in paragraphs]
            self._dense_index[paper_id] = list(self._embedder(texts)) if texts else []
        return self._dense_index[paper_id]

    def search(self, paper_id: str, query: str, limit: int | None = None) -> str:
        if limit is None:
            limit = self.search_limit
        if paper_id not in self.papers:
            return f"[error] {paper_id} not found"
        bm25, paragraphs, tokenized = self._index_for(paper_id)
        if bm25 is None:
            return "[no matching passages found]"
        from backend.retrieval import tokenize

        tokens = tokenize(query)
        if not tokens:
            return "[no matching passages found]"
        bm25_scores = bm25.get_scores(tokens)
        query_tokens = set(tokens)
        bm25_scores = [
            score + 0.5 * len(query_tokens & set(para_tokens))
            for score, para_tokens in zip(bm25_scores, tokenized)
        ]
        bm25_ranking = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranking = [i for i in bm25_ranking if bm25_scores[i] > 0]

        vectors = self._dense_vectors_for(paper_id)
        dense_ranking: list[int] = []
        if vectors:
            query_vector = self._embedder([query])[0]
            similarities = [_cosine_similarity(query_vector, v) for v in vectors]
            dense_ranking = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)

        if not bm25_ranking and not dense_ranking:
            return "[no matching passages found]"
        fused = reciprocal_rank_fusion([r for r in (bm25_ranking, dense_ranking) if r], k=self._rrf_k)
        lines = [self._format_passage(paragraphs[i]) for i in fused[:limit]]
        return "\n".join(lines) if lines else "[no matching passages found]"
