import json
import re
from collections import Counter
from pathlib import Path

from rank_bm25 import BM25Okapi

STOP = set("a an the is are was were be been to of for in on at as and or but with from that this it its how what why when which who does do did can could would should me about explain tell please compare versus between".split())
ALIASES = {
    "grpo": "group relative policy optimization",
    "ppo": "proximal policy optimization",
    "rlhf": "reinforcement learning human feedback",
    "强化学习": "reinforcement learning", "人类反馈": "human feedback rlhf",
    "奖励": "reward", "推理": "reasoning", "策略优化": "policy optimization",
    "组相对": "group relative grpo", "偏好": "preference dpo",
}


def tokenize(text: str) -> list[str]:
    for term, translation in ALIASES.items():
        text = text.replace(term, f" {translation} ")
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


class Corpus:
    def __init__(self, directory: Path, filename: str = "index.json"):
        self.directory = directory
        self.papers: list[dict] = []
        self.chunks: list[dict] = []
        self.index = None
        path = directory / filename
        if path.exists():
            data = json.loads(path.read_text())
            self.papers, self.chunks = data["papers"], data["chunks"]
            self.by_id = {p["id"]: p for p in self.papers}
            if self.chunks:
                self.index = BM25Okapi([
                    tokenize(c["quote"] + " " + self.by_id[c["paper_id"]]["title"])
                    for c in self.chunks
                ])

    def search(self, question: str, history: list[dict], limit: int = 8) -> list[dict]:
        if self.index is None:
            return []
        tokens = list(dict.fromkeys(tokenize(question)))
        scores = self.index.get_scores(tokens)
        # BM25 may assign zero/negative IDF to a term in a tiny corpus. A small
        # lexical overlap bonus keeps an exact acronym or title match searchable.
        scores += [0.5 * len(set(tokens) & set(tokenize(c["quote"] + " " + self.by_id[c["paper_id"]]["title"]))) for c in self.chunks]
        followup = re.search(r"\b(it|its|they|their|those|that|this|them|also)\b|它|上述|这个|刚才", question, re.I)
        if history and (followup or len(tokens) < 4):
            previous_questions = [m["content"] for m in history if m["role"] == "user"][-2:]
            previous_tokens = list(dict.fromkeys(tokenize(" ".join(previous_questions))))
            scores += 0.7 * self.index.get_scores(previous_tokens)
            scores += [0.5 * len(set(previous_tokens) & set(tokenize(c["quote"] + " " + self.by_id[c["paper_id"]]["title"]))) for c in self.chunks]
            # Helps follow-ups about a topic introduced by the last answer.
            last_answer = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
            answer_tokens = list(dict.fromkeys(tokenize(last_answer[:2000])))
            scores += 0.12 * self.index.get_scores(answer_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        selected, per_paper = [], Counter()
        for i in ranked:
            if scores[i] <= 0:
                break
            chunk = self.chunks[i]
            if per_paper[chunk["paper_id"]] >= 3:
                continue
            paper = self.by_id[chunk["paper_id"]]
            per_paper[chunk["paper_id"]] += 1
            selected.append({
                **chunk, "citation_id": f"S{len(selected) + 1}",
                "filename": paper.get("filename"), "title": paper["title"],
                "short_title": paper.get("short_title", paper["title"]), "url": paper.get("url"),
            })
            if len(selected) >= limit:
                break
        return selected
