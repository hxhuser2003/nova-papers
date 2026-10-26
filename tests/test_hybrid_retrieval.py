from pathlib import Path

import pytest

from backend.hybrid_retrieval import HybridQasperCorpus, reciprocal_rank_fusion


def test_rrf_favors_items_ranked_highly_by_multiple_rankers():
    # item 2 is first in one ranking and second in the other -- it should
    # beat item 0, which is only first in one ranking and absent from the other.
    fused = reciprocal_rank_fusion([[0, 1, 2], [2, 1]])
    assert fused[0] == 2


def test_rrf_handles_a_single_ranking():
    assert reciprocal_rank_fusion([[3, 1, 0]]) == [3, 1, 0]


def test_rrf_handles_no_rankings():
    assert reciprocal_rank_fusion([]) == []


def _write_corpus(tmp_path: Path) -> Path:
    import json

    papers = [{
        "paper_id": "p1", "title": "A Test Paper", "abstract": "We study group relative baselines.", "split": "train",
        "paragraphs": [
            {"para_id": "p_0", "section": "Intro", "text": "unrelated filler about weather patterns"},
            {"para_id": "p_1", "section": "Method", "text": "we use a group relative baseline for training"},
        ],
    }]
    (tmp_path / "papers.jsonl").write_text(json.dumps(papers[0]))
    (tmp_path / "questions.jsonl").write_text("")
    return tmp_path


# A tiny fake embedder: each text's vector is just its BM25-tokenized word
# set as a one-hot-ish bag, so cosine similarity behaves predictably without
# needing a real model -- this tests HybridQasperCorpus's wiring (fusion,
# formatting, error handling), not embedding quality itself.
def _bag_embedder(texts):
    from backend.retrieval import tokenize

    vocab = ["weather", "baseline", "training", "group", "relative", "filler"]
    return [[1.0 if word in tokenize(text) else 0.0 for word in vocab] for text in texts]


def test_search_fuses_bm25_and_dense_rankings(tmp_path):
    corpus = HybridQasperCorpus(_write_corpus(tmp_path), embedder=_bag_embedder)
    result = corpus.search("p1", "baseline")
    assert "[p_1]" in result
    assert result.index("[p_1]") < result.index("[p_0]") if "[p_0]" in result else True


def test_search_respects_limit(tmp_path):
    corpus = HybridQasperCorpus(_write_corpus(tmp_path), embedder=_bag_embedder)
    result = corpus.search("p1", "baseline training weather", limit=1)
    assert result.count("[p_") == 1


def test_search_unknown_paper_returns_the_same_error_as_the_base_class(tmp_path):
    corpus = HybridQasperCorpus(_write_corpus(tmp_path), embedder=_bag_embedder)
    assert corpus.search("missing", "x") == "[error] missing not found"


def test_search_with_no_query_tokens_reports_no_matches(tmp_path):
    corpus = HybridQasperCorpus(_write_corpus(tmp_path), embedder=_bag_embedder)
    assert corpus.search("p1", "") == "[no matching passages found]"


def test_read_and_find_paper_are_unchanged_from_the_base_class(tmp_path):
    corpus = HybridQasperCorpus(_write_corpus(tmp_path), embedder=_bag_embedder)
    assert corpus.read("p1", "p_1") == "[p_1] (Method) we use a group relative baseline for training"
    assert corpus.find_paper("group relative baseline") == "p1"
