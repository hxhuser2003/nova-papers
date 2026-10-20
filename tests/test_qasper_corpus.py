import json

from backend.qasper_corpus import QasperCorpus


def _write_fixture(tmp_path):
    papers = [
        {
            "paper_id": "p1", "title": "Paper One", "abstract": "...", "split": "train",
            "paragraphs": [
                {"para_id": "p_0", "section": "Intro", "text": "This paper studies clipped surrogate objectives for policy optimization."},
                {"para_id": "p_1", "section": "Method", "text": "We propose a group relative baseline instead of a learned value function."},
            ],
        },
        {
            "paper_id": "p2", "title": "Paper Two", "abstract": "...", "split": "validation",
            "paragraphs": [
                {"para_id": "p_0", "section": "Intro", "text": "Unrelated paper about speech recognition datasets."},
            ],
        },
    ]
    questions = [
        {
            "question_id": "q1", "paper_id": "p1", "question": "What baseline do they use instead of a value function?",
            "answers": [{"answer_type": "extractive", "answer_text": "group relative baseline",
                         "evidence_para_ids": ["p_1"], "has_non_text_evidence": False}],
        },
    ]
    (tmp_path / "papers.jsonl").write_text("\n".join(json.dumps(p) for p in papers))
    (tmp_path / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in questions))
    return tmp_path


def test_search_is_scoped_to_the_given_paper_only(tmp_path):
    corpus = QasperCorpus(_write_fixture(tmp_path))
    result = corpus.search("p1", "group relative baseline")
    assert "p_1" in result
    assert "speech recognition" not in result  # p2's content must never leak into p1's search


def test_search_on_unknown_paper_returns_an_error_string(tmp_path):
    corpus = QasperCorpus(_write_fixture(tmp_path))
    assert corpus.search("does-not-exist", "anything") == "[error] does-not-exist not found"


def test_search_with_no_match_says_so_instead_of_returning_nothing(tmp_path):
    corpus = QasperCorpus(_write_fixture(tmp_path))
    assert corpus.search("p1", "zzz nonexistent query terms") == "[no matching passages found]"


def test_read_returns_full_paragraph_text_with_section(tmp_path):
    corpus = QasperCorpus(_write_fixture(tmp_path))
    result = corpus.read("p1", "p_1")
    assert result == "[p_1] (Method) We propose a group relative baseline instead of a learned value function."


def test_read_on_unknown_para_id_returns_an_error_string(tmp_path):
    corpus = QasperCorpus(_write_fixture(tmp_path))
    assert corpus.read("p1", "p_99") == "[error] p_99 not found"


def test_read_truncation_only_applies_to_search_not_read(tmp_path):
    tmp_path = _write_fixture(tmp_path)
    data = json.loads((tmp_path / "papers.jsonl").read_text().splitlines()[0])
    data["paragraphs"].append({"para_id": "p_2", "section": "Results", "text": " ".join(f"word{i}" for i in range(200))})
    lines = (tmp_path / "papers.jsonl").read_text().splitlines()
    lines[0] = json.dumps(data)
    (tmp_path / "papers.jsonl").write_text("\n".join(lines))

    corpus = QasperCorpus(tmp_path)
    assert "word199" in corpus.read("p1", "p_2")  # read returns the full paragraph, untruncated


def test_questions_for_split_filters_by_the_papers_split(tmp_path):
    corpus = QasperCorpus(_write_fixture(tmp_path))
    assert [q["question_id"] for q in corpus.questions_for_split("train")] == ["q1"]
    assert corpus.questions_for_split("validation") == []


def _write_catalog_fixture(tmp_path):
    papers = [
        {"paper_id": "ppo", "title": "Proximal Policy Optimization Algorithms",
         "abstract": "We propose a clipped surrogate objective for policy gradient methods.",
         "split": "train", "paragraphs": []},
        {"paper_id": "speech", "title": "Speech Recognition with Deep Neural Networks",
         "abstract": "We study acoustic models for transcribing spoken audio into text.",
         "split": "train", "paragraphs": []},
    ]
    (tmp_path / "papers.jsonl").write_text("\n".join(json.dumps(p) for p in papers))
    (tmp_path / "questions.jsonl").write_text("")
    return tmp_path


def test_find_paper_matches_the_most_relevant_title_and_abstract(tmp_path):
    corpus = QasperCorpus(_write_catalog_fixture(tmp_path))
    assert corpus.find_paper("How does the clipped surrogate objective work in policy optimization?") == "ppo"
    assert corpus.find_paper("How are acoustic models used for speech transcription?") == "speech"


def test_find_paper_returns_none_when_nothing_matches(tmp_path):
    corpus = QasperCorpus(_write_catalog_fixture(tmp_path))
    assert corpus.find_paper("zzz nonexistent unrelated query") is None
