"""从 QASPER 预处理数据中挑选样题（任务 1：示例题集）。

用法（在仓库根目录运行）：
    python scripts/select_sample_questions.py

可选参数：
    --data-dir corpus/qasper      papers.jsonl / questions.jsonl 所在目录
    --out-dir eval                输出目录
    --split validation            从哪个划分挑题
    --seed 42                     随机种子，保证每次挑出来的一样
    --pool-factor 2               候选池是最终题数的几倍（供人工审核时替换）

输出：
    eval/sample_questions.jsonl   最终样题（默认 40 道），基线测试直接读它
    eval/review_candidates.csv    候选池，含论文标题、答案、证据原文，
                                  用 Excel / Google Sheets 打开人工审核
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# 每类题目的数量，加起来是最终样题数
QUOTAS = {
    "extractive": 15,
    "free_form": 10,
    "yes_no": 10,
    "unanswerable": 5,
}
MAX_PER_PAPER = 2           # 同一篇论文最多挑几道，保证覆盖面
MULTI_EVIDENCE_RATIO = 0.5  # 每类里尽量有一半是多段证据题


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evidence_union(question):
    ids = set()
    for ans in question["answers"]:
        ids.update(ans.get("evidence_para_ids", []))
    return sorted(ids, key=lambda x: int(x.split("_")[1]))


def classify(question):
    """返回 (题型, 不合格原因)。原因为 None 表示可用。"""
    answers = question["answers"]
    if not answers:
        return None, "没有参考答案"
    if any(a.get("has_non_text_evidence") for a in answers):
        return None, "涉及图表证据"
    types = {a["answer_type"] for a in answers}
    if len(types) > 1:
        # 例如一位标注者答 Yes、另一位标 unanswerable，题目本身有歧义
        return None, "标注者答案类型不一致"
    qtype = types.pop()
    if qtype == "yes_no":
        verdicts = {a.get("answer_text", "").strip().lower() for a in answers}
        if len(verdicts) > 1:
            # 例如一位答 Yes、另一位答 No
            return None, "是非题标注者答案矛盾"
    if qtype != "unanswerable":
        if any(not a.get("evidence_para_ids") for a in answers):
            return None, "缺少证据段落"
        if any(not a.get("answer_text", "").strip() for a in answers):
            return None, "答案文本为空"
    return qtype, None


def auto_reason(qtype, n_evidence, n_answers):
    parts = {
        "extractive": "抽取式，答案可直接在原文中定位",
        "free_form": "自由回答，需要概括原文",
        "yes_no": "是非题，考查能否找到支持证据",
        "unanswerable": "无法回答题，考查模型能否拒答而不是编造",
    }
    reason = parts[qtype]
    if n_evidence >= 2:
        reason += f"；证据分布在 {n_evidence} 段，适合检验多轮搜索"
    if n_answers >= 2:
        reason += f"；有 {n_answers} 位标注者答案"
    return reason


def pick(candidates, n, rng):
    """按配额挑题：多段证据题优先占一半，同一论文不超过 MAX_PER_PAPER。"""
    multi = [q for q in candidates if q["_n_evidence"] >= 2]
    single = [q for q in candidates if q["_n_evidence"] < 2]
    rng.shuffle(multi)
    rng.shuffle(single)
    target_multi = round(n * MULTI_EVIDENCE_RATIO)
    ordered = multi[:target_multi] + single + multi[target_multi:]

    chosen, per_paper = [], Counter()
    for q in ordered:
        if len(chosen) >= n:
            break
        if per_paper[q["paper_id"]] >= MAX_PER_PAPER:
            continue
        chosen.append(q)
        per_paper[q["paper_id"]] += 1
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="corpus/qasper")
    parser.add_argument("--out-dir", default="eval")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-factor", type=int, default=2)
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    papers = {p["paper_id"]: p for p in load_jsonl(data_dir / "papers.jsonl")}
    questions = load_jsonl(data_dir / "questions.jsonl")

    # split 挂在论文上，需要通过 paper_id 关联
    in_split = [q for q in questions
                if papers.get(q["paper_id"], {}).get("split") == args.split]

    by_type, rejected = defaultdict(list), Counter()
    for q in in_split:
        qtype, reason = classify(q)
        if reason:
            rejected[reason] += 1
            continue
        q["_type"] = qtype
        q["_evidence"] = evidence_union(q)
        q["_n_evidence"] = len(q["_evidence"])
        by_type[qtype].append(q)

    print(f"{args.split} 共 {len(in_split)} 道题")
    for reason, n in rejected.most_common():
        print(f"  排除 {n:4d} 道：{reason}")
    for t in QUOTAS:
        n_multi = sum(q["_n_evidence"] >= 2 for q in by_type[t])
        print(f"  可用 {t:12s} {len(by_type[t]):4d} 道（多段证据 {n_multi} 道）")

    final, pool = [], []
    for qtype, quota in QUOTAS.items():
        chosen = pick(by_type[qtype], quota * args.pool_factor, rng)
        if len(chosen) < quota:
            print(f"警告：{qtype} 只有 {len(chosen)} 道可用，少于配额 {quota}")
        for i, q in enumerate(chosen):
            q["_selected"] = i < quota  # 前 quota 道进最终样题，其余作为备选
        final.extend(chosen[:quota])
        pool.extend(chosen)

    # 最终样题：基线测试直接读这个文件
    with open(out_dir / "sample_questions.jsonl", "w", encoding="utf-8") as f:
        for q in final:
            record = {
                "question_id": q["question_id"],
                "paper_id": q["paper_id"],
                "question": q["question"],
                "answer_type": q["_type"],
                "answers": q["answers"],
                "evidence_para_ids": q["_evidence"],
                "reason": auto_reason(q["_type"], q["_n_evidence"], len(q["answers"])),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 审核表：带上证据原文，方便人工判断题目是否合适
    with open(out_dir / "review_candidates.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["selected", "answer_type", "question_id", "paper_id", "paper_title",
                         "question", "answers", "evidence_para_ids", "evidence_text",
                         "reason", "keep(人工填 Y/N)", "note"])
        for q in pool:
            paras = {p["para_id"]: p for p in papers[q["paper_id"]]["paragraphs"]}
            evidence_text = "\n\n".join(
                f"[{pid}] ({paras[pid]['section']}) {paras[pid]['text']}"
                for pid in q["_evidence"] if pid in paras)
            answers = " || ".join(a["answer_text"] or "(unanswerable)" for a in q["answers"])
            writer.writerow([
                "Y" if q["_selected"] else "backup", q["_type"], q["question_id"],
                q["paper_id"], papers[q["paper_id"]]["title"], q["question"], answers,
                ", ".join(q["_evidence"]), evidence_text,
                auto_reason(q["_type"], q["_n_evidence"], len(q["answers"])), "", "",
            ])

    print(f"\n已写入 {out_dir / 'sample_questions.jsonl'}（{len(final)} 道）")
    print(f"已写入 {out_dir / 'review_candidates.csv'}（候选 {len(pool)} 道，含备选）")


if __name__ == "__main__":
    main()
