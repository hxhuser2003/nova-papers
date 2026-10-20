"""训练前基线测试（任务 3A）。

在 eval/sample_questions.jsonl 的样题上跑三组基线，计算 QASPER 官方口径的
Answer F1 / Evidence F1：

    direct : 只给标题 + 摘要 + 问题，直接作答（不检索）
    rag    : 用问题检索一次（BM25 top-3），把结果连同问题交给模型作答
    agent  : 用 backend/agent_loop.py 的多轮 Agent 循环（search / read / answer）

前提：vLLM 服务已在 localhost:8000 启动（Qwen/Qwen2.5-3B-Instruct）。

用法（在仓库根目录运行）：
    python scripts/run_baseline.py                     # 三组都跑
    python scripts/run_baseline.py --modes agent       # 只跑 agent
输出：
    eval/results/<mode>.jsonl   每道题的预测、得分、轨迹
    eval/results/summary.md     汇总表
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.qasper_metrics import answer_f1, evidence_f1, gold_text  # noqa: E402

STOPS = ["</search>", "</read>", "</answer>", "<information>"]
ACTION_ENDS = ("</search>", "</read>", "</answer>")
PARA_ID = re.compile(r"\[(p_\d+)\]")
ANSWER_TAG = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.S)


# ---------------------------------------------------------------- 模型客户端

class VLLMClient:
    """把 vLLM 服务包装成 agent_loop 需要的 ModelClient：messages(list[dict]) -> 本轮输出(str)。

    直接转发给 /v1/chat/completions，chat template 由 vLLM 在服务端套用，
    不需要本地再装 tokenizer 自己拼 prompt（也因此不再需要 transformers 依赖）。
    """

    def __init__(self, base_url: str, model: str, max_tokens: int, temperature: float):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _complete_chat(self, messages: list[dict]) -> str:
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=self.max_tokens,
            temperature=self.temperature, stop=STOPS,
            # vLLM 专有扩展参数（标准 OpenAI API 没有），要求服务端把匹配到的
            # 停止符保留在输出里而不是去掉，这样循环解析时标签才是完整的。
            extra_body={"include_stop_str_in_output": True},
        )
        choice = r.choices[0]
        text = choice.message.content
        stop = getattr(choice, "stop_reason", None)
        if choice.finish_reason == "stop" and stop in ACTION_ENDS and not text.endswith(stop):
            text += stop  # 停止符本身不会出现在输出里，补回来让循环能解析
        return text

    def __call__(self, messages: list[dict]) -> str:
        return self._complete_chat(messages)

    def chat(self, user: str) -> str:
        return self._complete_chat([{"role": "user", "content": user}])


# ---------------------------------------------------------------- 三组基线

SINGLE_SHOT_RULES = (
    "Answer rules: for yes/no questions answer only \"Yes\" or \"No\"; if the answer cannot be "
    "determined, answer \"Unanswerable\"; otherwise answer briefly, using wording from the paper. "
    "INLINEFORM/DISPLAYFORM are removed formulas and BIBREF/TABREF/FIGREF are citation markers; "
    "never copy them into your answer.\n"
    "First reason inside <think>...</think>, then give the final answer inside <answer>...</answer>."
)


def extract_answer(text: str) -> str | None:
    m = ANSWER_TAG.search(text)
    if m:
        return m.group(1).strip()
    return None


def run_direct(q, paper, corpus, client):
    user = (
        f"You are answering a question about an NLP research paper. You only have its title and abstract.\n\n"
        f"Paper title: {paper['title']}\nAbstract: {paper['abstract']}\n\n"
        f"Question: {q['question']}\n\n{SINGLE_SHOT_RULES}"
    )
    t = time.monotonic()
    out = client.chat(user)
    return {"output": out, "prediction": extract_answer(out), "evidence": [],
            "latency_s": time.monotonic() - t}


def run_rag(q, paper, corpus, client):
    retrieved = corpus.search(q["paper_id"], q["question"])
    user = (
        f"You are answering a question about an NLP research paper.\n\n"
        f"Paper title: {paper['title']}\nAbstract: {paper['abstract']}\n\n"
        f"Relevant paragraphs retrieved from the paper:\n{retrieved}\n\n"
        f"Question: {q['question']}\n\n{SINGLE_SHOT_RULES}"
    )
    t = time.monotonic()
    out = client.chat(user)
    return {"output": out, "prediction": extract_answer(out),
            "evidence": sorted(set(PARA_ID.findall(retrieved))),
            "latency_s": time.monotonic() - t}


def run_agent(q, paper, corpus, client):
    from backend.agent_loop import run_episode

    t = time.monotonic()
    traj = run_episode(q["question_id"], q["question"], q["paper_id"], corpus, client)
    latency = time.monotonic() - t
    read_ids, seen_ids = set(), set()
    for step in traj.steps:
        action = step.action
        if getattr(action, "kind", None) == "read":
            read_ids.add(action.content.strip())
        if step.tool_result:
            seen_ids.update(PARA_ID.findall(step.tool_result))
    steps = [asdict(s) if is_dataclass(s) else str(s) for s in traj.steps]
    return {
        "prediction": traj.final_answer,
        "evidence": sorted(read_ids),           # 口径一：模型 read 过的段落
        "evidence_seen": sorted(seen_ids),      # 口径二：模型看到过的所有段落（search 结果 + read）
        "terminated_reason": traj.terminated_reason,
        "rounds": len(traj.steps),
        "latency_s": latency,
        "steps": steps,
    }


MODES = {"direct": run_direct, "rag": run_rag, "agent": run_agent}


# ---------------------------------------------------------------- 主流程

def summarize(mode: str, records: list[dict]) -> str:
    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    lines = [f"### {mode}", "",
             "| 题型 | 题数 | Answer F1 | Evidence F1 |", "| --- | --- | --- | --- |"]
    by_type = defaultdict(list)
    for r in records:
        by_type[r["answer_type"]].append(r)
    for t in ["extractive", "free_form", "yes_no", "unanswerable"]:
        rs = by_type.get(t, [])
        lines.append(f"| {t} | {len(rs)} | {mean(r['answer_f1'] for r in rs):.3f} | "
                     f"{mean(r['evidence_f1'] for r in rs):.3f} |")
    lines.append(f"| **全部** | {len(records)} | **{mean(r['answer_f1'] for r in records):.3f}** | "
                 f"**{mean(r['evidence_f1'] for r in records):.3f}** |")
    lines.append("")
    lines.append(f"- 有答案（成功解析出 answer）：{sum(r['prediction'] is not None for r in records)}/{len(records)}")
    lines.append(f"- 平均耗时：{mean(r['latency_s'] for r in records):.1f} 秒/题")
    if mode == "agent":
        reasons = Counter(r["terminated_reason"] for r in records)
        lines.append(f"- 结束原因：{dict(reasons)}")
        lines.append(f"- 平均轮数：{mean(r['rounds'] for r in records):.2f}")
        lines.append(f"- 至少 read 过一次：{sum(bool(r['evidence']) for r in records)}/{len(records)}")
        lines.append(f"- Evidence F1（口径二，看到过的所有段落）："
                     f"{mean(r['evidence_f1_seen'] for r in records):.3f}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["direct", "rag", "agent"], choices=list(MODES))
    parser.add_argument("--samples", default="eval/sample_questions.jsonl")
    parser.add_argument("--corpus", default="corpus/qasper")
    parser.add_argument("--prompt", default="eval/system_prompt.txt",
                        help="替换 agent_loop 的 SYSTEM_PROMPT；传空字符串则用 agent_loop 自带的")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out-dir", default="eval/results")
    args = parser.parse_args()

    from backend import agent_loop
    from backend.qasper_corpus import QasperCorpus

    if args.prompt:
        agent_loop.SYSTEM_PROMPT = Path(args.prompt).read_text(encoding="utf-8").strip()
        print(f"agent 使用的系统提示词：{args.prompt}")

    corpus = QasperCorpus(Path(args.corpus))
    client = VLLMClient(args.base_url, args.model, args.max_tokens, args.temperature)
    samples = [json.loads(l) for l in open(args.samples, encoding="utf-8") if l.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = ["# 训练前基线测试结果", "",
              f"- 模型：{args.model}（vLLM，temperature={args.temperature}）",
              f"- 样题：{args.samples}（{len(samples)} 道）", ""]

    for mode in args.modes:
        fn = MODES[mode]

        def one(q):
            paper = corpus.papers[q["paper_id"]]
            try:
                res = fn(q, paper, corpus, client)
            except Exception as e:  # 单题出错不影响整组
                res = {"prediction": None, "evidence": [], "latency_s": 0.0, "error": repr(e),
                       "terminated_reason": "exception", "rounds": 0}
            res.update({
                "question_id": q["question_id"], "paper_id": q["paper_id"],
                "question": q["question"], "answer_type": q["answer_type"],
                "gold": [gold_text(a) for a in q["answers"]],
            })
            res["answer_f1"] = answer_f1(res["prediction"], q["answers"])
            res["evidence_f1"] = evidence_f1(set(res["evidence"]), q["answers"])
            if mode == "agent":
                res["evidence_f1_seen"] = evidence_f1(set(res.get("evidence_seen", [])), q["answers"])
            return res

        t = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(one, samples))
        errors = [r for r in records if "error" in r]
        print(f"[{mode}] 完成 {len(records)} 道，用时 {time.time() - t:.0f} 秒，出错 {len(errors)} 道")
        for r in errors[:3]:
            print("   ", r["question_id"], r["error"])

        with open(out_dir / f"{mode}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        section = summarize(mode, records)
        print(section)
        report.append(section)

    (out_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"汇总已写入 {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
