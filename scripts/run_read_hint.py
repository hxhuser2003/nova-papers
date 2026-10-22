"""Evaluation-only experiment: hint only on genuinely truncated passages."""
from backend.qasper_corpus import QasperCorpus, MAX_PASSAGE_WORDS
from scripts.run_baseline import main

_original_format = QasperCorpus._format_passage


def hinted_passage(paragraph, truncate=True):
    result = _original_format(paragraph, truncate=truncate)
    if truncate and len(paragraph['text'].split()) > MAX_PASSAGE_WORDS:
        result += f"\n[This passage is truncated. Use <read>{paragraph['para_id']}</read> to view the full text.]"
    return result


if __name__ == '__main__':
    QasperCorpus._format_passage = staticmethod(hinted_passage)
    main()
