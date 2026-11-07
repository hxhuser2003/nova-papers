"""R10 (inference-only): swap QasperCorpus for HybridQasperCorpus before
running the normal baseline eval. Model/prompt/agent_loop untouched --
only which corpus class run_baseline.main() constructs changes."""
from __future__ import annotations

import os

import backend.qasper_corpus as qasper_corpus
from backend.hybrid_retrieval import HybridQasperCorpus, SentenceTransformerEmbedder
from scripts.run_baseline import main


def _hybrid_corpus(directory, search_limit=None):
    # Constructed here, not at module import time: loading the real
    # embedding model must not happen just because something imported this
    # module (e.g. a test importing a sibling symbol) -- it should only
    # happen when a corpus is actually built for a real run.
    #
    # EMBEDDER_DEVICE defaults to "cpu" (the prior hardcoded behavior) --
    # this ran alongside vLLM at --gpu-memory-utilization 0.80, which can
    # leave little headroom for a second model on the same GPU. Set
    # EMBEDDER_DEVICE=cuda to try moving BAAI/bge-m3 onto the GPU instead;
    # this is the suspected main cause of R10's measured 53x latency
    # penalty (6.6s -> 351.1s/question, see docs/EXPERIMENT_RECORD.md), but
    # has not actually been tried yet -- verify it doesn't OOM before
    # trusting it for a real comparison run.
    device = os.environ.get("EMBEDDER_DEVICE", "cpu")
    # Printed so a job's log confirms what actually ran, instead of trusting
    # that sbatch's default --export=ALL carried this env var through --
    # that default isn't something this project has verified on CARC, and a
    # silent fallback to "cpu" would look identical to a real failed attempt
    # (both just measure the same ~351s/question latency, no error raised).
    print(f"HybridQasperCorpus embedder device: {device}", flush=True)
    kwargs = {} if search_limit is None else {"search_limit": search_limit}
    return HybridQasperCorpus(directory, embedder=SentenceTransformerEmbedder(device=device), **kwargs)


if __name__ == "__main__":
    qasper_corpus.QasperCorpus = _hybrid_corpus
    main()
