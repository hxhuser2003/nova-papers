"""根据人工审核结果生成最终样题集。

用法（在仓库根目录运行）：
    python scripts/finalize_sample_questions.py

读取：
    eval/review_final.csv          人工审核结果（keep = Y 的题进入样题集）
    corpus/qasper/questions.jsonl  取完整的参考答案（含每个答案各自的证据）
输出：
    eval/sample_questions.jsonl    最终样题，基线测试直接读这个文件
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", default="eval/review_final.csv")
    parser.add_argument("--questions", default="corpus/qasper/questions.jsonl")
    parser.add_argument("--out", default="eval/sample_questions.jsonl")
    args = parser.parse_args()

    with open(args.review, encoding="utf-8-sig") as f:
        kept = [r for r in csv.DictReader(f) if r["keep"].strip().upper() == "Y"]

    wanted = {r["question_id"] for r in kept}
    questions = {}
    with open(args.questions, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["question_id"] in wanted:
                questions[q["question_id"]] = q

    missing = wanted - questions.keys()
    if missing:
        raise SystemExit(f"questions.jsonl 中找不到这些题：{sorted(missing)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            q = questions[r["question_id"]]
            evidence = sorted(
                {pid for a in q["answers"] for pid in a.get("evidence_para_ids", [])},
                key=lambda x: int(x.split("_")[1]),
            )
            record = {
                "question_id": q["question_id"],
                "paper_id": q["paper_id"],
                "question": q["question"],
                "answer_type": r["answer_type"],
                "answers": q["answers"],
                "evidence_para_ids": evidence,
                "multi_evidence": len(evidence) >= 2,
                "note": r["note"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = Counter(r["answer_type"] for r in kept)
    print(f"已写入 {args.out}：共 {len(kept)} 道")
    for t in ["extractive", "free_form", "yes_no", "unanswerable"]:
        print(f"  {t:12s} {counts[t]}")


if __name__ == "__main__":
    main()
