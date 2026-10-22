from backend.qasper_corpus import QasperCorpus, MAX_PASSAGE_WORDS
from scripts.run_read_hint import hinted_passage

def test_hint_only_on_truncated_search_results():
    p = dict(para_id='p_0', section='Methods', text=' '.join(['word'] * (MAX_PASSAGE_WORDS + 1)))
    original = QasperCorpus._format_passage(p)
    assert hinted_passage(p).startswith(original)
    assert '<read>p_0</read>' in hinted_passage(p)
    assert hinted_passage(p, truncate=False) == QasperCorpus._format_passage(p, truncate=False)
    p['text'] = 'short text'
    assert hinted_passage(p) == QasperCorpus._format_passage(p)
