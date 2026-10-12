"""Paper-scoped data access for QASPER, backing the Agent loop's search/read
tools (see project1's "Tool Requirements and Action Format v0.1" spec) and
paper/question lookup for baseline testing and training.

Retrieval is deliberately scoped to one paper at a time (not the whole
corpus): the spec restricts the agent to searching within the paper it was
given, so a fresh small BM25 index is built per paper on first use and
cached, rather than reusing backend.retrieval.Corpus's single global index.
"""

from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.retrieval import tokenize

SEARCH_LIMIT = 3
MAX_PASSAGE_WORDS = 150


class QasperCorpus:
    def __init__(self, directory: Path):
        self.papers: dict[str, dict] = {}
        self.questions: dict[str, dict] = {}
        self._paper_index: dict[str, tuple] = {}

        for line in (directory / "papers.jsonl").read_text().splitlines():
            if line.strip():
                paper = json.loads(line)
                self.papers[paper["paper_id"]] = paper
        for line in (directory / "questions.jsonl").read_text().splitlines():
            if line.strip():
                question = json.loads(line)
                self.questions[question["question_id"]] = question

    def _index_for(self, paper_id: str):
        if paper_id not in self._paper_index:
            paragraphs = self.papers[paper_id]["paragraphs"]
            tokenized = [tokenize(p["text"]) for p in paragraphs]
            bm25 = BM25Okapi(tokenized) if paragraphs else None
            self._paper_index[paper_id] = (bm25, paragraphs, tokenized)
        return self._paper_index[paper_id]

    def search(self, paper_id: str, query: str, limit: int = SEARCH_LIMIT) -> str:
        """The <search> tool: up to `limit` passages from this paper only,
        each truncated to ~MAX_PASSAGE_WORDS words, formatted for the
        <information> block as "[para_id] (section) text"."""
        if paper_id not in self.papers:
            return f"[error] {paper_id} not found"
        bm25, paragraphs, tokenized = self._index_for(paper_id)
        tokens = tokenize(query)
        if bm25 is None or not tokens:
            return "[no matching passages found]"
        scores = bm25.get_scores(tokens)
        # A single paper is a tiny BM25 corpus (tens of paragraphs), where IDF
        # can collapse to zero for every term (see backend.retrieval.Corpus for
        # the same issue at corpus scale). A lexical overlap bonus keeps an
        # exact term match searchable even when BM25 alone scores everything 0.
        query_tokens = set(tokens)
        scores = [
            score + 0.5 * len(query_tokens & set(para_tokens))
            for score, para_tokens in zip(scores, tokenized)
        ]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        lines = []
        for i in ranked[:limit]:
            if scores[i] <= 0:
                break
            lines.append(self._format_passage(paragraphs[i]))
        return "\n".join(lines) if lines else "[no matching passages found]"

    def read(self, paper_id: str, para_id: str) -> str:
        """The <read> tool: full text of one paragraph."""
        paper = self.papers.get(paper_id)
        if paper is None:
            return f"[error] {paper_id} not found"
        for p in paper["paragraphs"]:
            if p["para_id"] == para_id:
                return self._format_passage(p, truncate=False)
        return f"[error] {para_id} not found"

    def questions_for_split(self, split: str) -> list[dict]:
        """All questions whose paper belongs to `split` ('train'/'validation'/'test')."""
        return [q for q in self.questions.values() if self.papers[q["paper_id"]]["split"] == split]

    @staticmethod
    def _format_passage(paragraph: dict, truncate: bool = True) -> str:
        text = paragraph["text"]
        if truncate:
            words = text.split()
            if len(words) > MAX_PASSAGE_WORDS:
                text = " ".join(words[:MAX_PASSAGE_WORDS]) + " …"
        return f"[{paragraph['para_id']}] ({paragraph['section']}) {text}"
