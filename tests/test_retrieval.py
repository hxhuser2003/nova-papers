import json

from backend.retrieval import Corpus


def test_retrieval_returns_exact_passages_and_balances_papers(tmp_path):
    data = {
        "papers": [
            {"id": "ppo", "filename": "ppo.pdf", "title": "Proximal Policy Optimization", "short_title": "PPO", "url": "https://arxiv.org/abs/1707.06347"},
            {"id": "math", "filename": "math.pdf", "title": "Group Relative Policy Optimization", "short_title": "DeepSeekMath", "url": "https://arxiv.org/abs/2402.03300"},
        ],
        "chunks": [
            {"id": "a", "paper_id": "ppo", "page": 1, "quote": "PPO uses a clipped surrogate objective to constrain policy updates."},
            {"id": "b", "paper_id": "math", "page": 2, "quote": "GRPO samples a group of outputs and uses relative rewards within the group."},
        ],
    }
    (tmp_path / "index.json").write_text(json.dumps(data))
    corpus = Corpus(tmp_path)
    sources = corpus.search("How does GRPO differ from PPO?", [])
    assert sources
    assert {source["filename"] for source in sources} == {"math.pdf", "ppo.pdf"}
    assert all(source["quote"] in {chunk["quote"] for chunk in data["chunks"]} for source in sources)


def test_follow_up_includes_previous_user_topic(tmp_path):
    data = {
        "papers": [{"id": "ppo", "filename": "ppo.pdf", "title": "Proximal Policy Optimization", "short_title": "PPO", "url": "https://arxiv.org/abs/x"}],
        "chunks": [{"id": "a", "paper_id": "ppo", "page": 1, "quote": "PPO clips the probability ratio in the surrogate objective."}],
    }
    (tmp_path / "index.json").write_text(json.dumps(data))
    corpus = Corpus(tmp_path)
    sources = corpus.search("How does it work?", [{"role": "user", "content": "Explain PPO"}])
    assert sources[0]["paper_id"] == "ppo"
