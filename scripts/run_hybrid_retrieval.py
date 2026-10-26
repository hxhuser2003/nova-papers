"""R10 (inference-only): swap QasperCorpus for HybridQasperCorpus before
running the normal baseline eval. Model/prompt/agent_loop untouched --
only which corpus class run_baseline.main() constructs changes."""
from __future__ import annotations

import backend.qasper_corpus as qasper_corpus
from backend.hybrid_retrieval import HybridQasperCorpus, SentenceTransformerEmbedder
from scripts.run_baseline import main


def _hybrid_corpus(directory):
    # Constructed here, not at module import time: loading the real
    # embedding model must not happen just because something imported this
    # module (e.g. a test importing a sibling symbol) -- it should only
    # happen when a corpus is actually built for a real run.
    return HybridQasperCorpus(directory, embedder=SentenceTransformerEmbedder())


if __name__ == "__main__":
    qasper_corpus.QasperCorpus = _hybrid_corpus
    main()
