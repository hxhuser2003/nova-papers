"""Download QASPER (via the HF datasets-server REST API, no `datasets`/pyarrow
dependency needed) and convert it into two JSONL files consumed by three
downstream pieces: the search/read tools (paper_id + para_id lookup), the
reward function (Evidence F1 / Answer F1), and the train/val/test split
(kept at the paper level so no paper's questions cross splits).

Output: corpus/qasper/papers.jsonl, corpus/qasper/questions.jsonl

papers.jsonl (one paper per line):
  {paper_id, title, abstract, split, paragraphs: [{para_id, section, text}]}
  para_id is "p_N", flattened and globally sequential across the whole paper
  (not reset per section), so <read>p_N</read> is unambiguous.

questions.jsonl (one question per line):
  {question_id, paper_id, question, answers: [
      {answer_type, answer_text, evidence_para_ids, has_non_text_evidence}
  ]}
  answer_type is normalized to exactly one of:
  extractive | yes_no | free_form | unanswerable
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "corpus" / "qasper"
API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100


def fetch_papers(split: str, limit: int | None, client: httpx.Client) -> list[dict]:
    offset = 0
    papers = []
    while limit is None or len(papers) < limit:
        length = PAGE_SIZE if limit is None else min(PAGE_SIZE, limit - len(papers))
        resp = client.get(API, params={
            "dataset": "allenai/qasper", "config": "qasper", "split": split,
            "offset": offset, "length": length,
        })
        resp.raise_for_status()
        rows = resp.json()["rows"]
        if not rows:
            break
        papers.extend(r["row"] for r in rows)
        offset += len(rows)
        print(f"  fetched {len(papers)} papers…", flush=True)
        if len(rows) < length:
            break
        time.sleep(0.2)
    return papers


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_paragraphs(paper: dict) -> list[dict]:
    """Flattened, globally sequential para_id across the whole paper (not
    reset per section) so <read>p_N</read> never collides across sections."""
    paragraphs = []
    full_text = paper["full_text"]
    for section_name, para_list in zip(full_text["section_name"], full_text["paragraphs"]):
        for para in para_list:
            text = normalize(para)
            if len(text) < 20:
                continue
            paragraphs.append({
                "para_id": f"p_{len(paragraphs)}",
                "section": section_name or "",
                "text": text,
            })
    return paragraphs


def match_evidence(evidence_text: str, paragraphs: list[dict]) -> tuple[str | None, bool, bool]:
    """Returns (para_id or None, has_non_text_evidence, is_ambiguous).
    A FLOAT_SELECTED reference (table/figure) is reported distinctly from a
    genuine, unexpected mismatch so the two failure modes aren't conflated.

    Exact-text matches are checked before substring matches so that a
    coincidental substring hit earlier in the paper never wins over a real
    exact match later in it (papers can repeat a sentence verbatim, e.g. in
    an intro and a conclusion). is_ambiguous is True when more than one
    paragraph is an equally good match (e.g. a duplicated paragraph) and the
    choice of the first one is therefore not fully reliable."""
    norm_ev = normalize(evidence_text)
    if not norm_ev:
        return None, False, False
    if norm_ev.startswith("FLOAT SELECTED"):
        return None, True, False
    exact = [para["para_id"] for para in paragraphs if norm_ev == para["text"]]
    if exact:
        return exact[0], False, len(exact) > 1
    partial = [para["para_id"] for para in paragraphs if norm_ev in para["text"] or para["text"] in norm_ev]
    if partial:
        return partial[0], False, len(partial) > 1
    return None, False, False


def normalize_answer(ans: dict) -> dict:
    """QASPER's four raw fields -> exactly one of extractive/yes_no/free_form/unanswerable."""
    if ans["unanswerable"]:
        answer_type, answer_text = "unanswerable", ""
    elif ans["yes_no"] is not None:
        answer_type, answer_text = "yes_no", ("Yes" if ans["yes_no"] else "No")
    elif ans["extractive_spans"]:
        answer_type, answer_text = "extractive", " / ".join(ans["extractive_spans"])
    elif ans["free_form_answer"].strip():
        answer_type, answer_text = "free_form", ans["free_form_answer"].strip()
    else:
        raise ValueError(f"No answer field set and not marked unanswerable: {ans}")
    return {"answer_type": answer_type, "answer_text": answer_text}


def build_questions(paper: dict, paragraphs: list[dict]) -> tuple[list[dict], int]:
    questions, ambiguous_count = [], 0
    qas = paper["qas"]
    for question, question_id, answer_group in zip(qas["question"], qas["question_id"], qas["answers"]):
        answers = []
        for ans in answer_group["answer"]:
            normalized = normalize_answer(ans)
            evidence_para_ids, has_non_text_evidence = [], False
            for ev in ans["evidence"]:
                para_id, is_float, is_ambiguous = match_evidence(ev, paragraphs)
                has_non_text_evidence = has_non_text_evidence or is_float
                ambiguous_count += is_ambiguous
                if para_id is not None:
                    evidence_para_ids.append(para_id)
            answers.append({
                **normalized,
                "evidence_para_ids": evidence_para_ids,
                "has_non_text_evidence": has_non_text_evidence,
            })
        questions.append({
            "question_id": question_id, "paper_id": paper["id"],
            "question": question, "answers": answers,
        })
    return questions, ambiguous_count


def process_split(split: str, limit: int | None, client: httpx.Client, papers_out: list, questions_out: list):
    print(f"Fetching {split} (limit={limit})…", flush=True)
    raw_papers = fetch_papers(split, limit, client)

    total_questions = matched_evidence = float_evidence = ambiguous_evidence = 0
    for paper in raw_papers:
        paragraphs = build_paragraphs(paper)
        questions, ambiguous_count = build_questions(paper, paragraphs)
        total_questions += len(questions)
        ambiguous_evidence += ambiguous_count

        for q in questions:
            for ans in q["answers"]:
                matched_evidence += len(ans["evidence_para_ids"])
            if any(a["has_non_text_evidence"] for a in q["answers"]):
                float_evidence += 1

        papers_out.append({
            "paper_id": paper["id"], "title": paper["title"], "abstract": paper["abstract"],
            "split": split, "paragraphs": paragraphs,
        })
        questions_out.extend(questions)

    print(f"  {split}: {len(raw_papers)} papers, {total_questions} questions, "
          f"{matched_evidence} evidence spans matched, {float_evidence} questions touch float evidence, "
          f"{ambiguous_evidence} evidence spans matched more than one paragraph equally well")


def main():
    papers_out, questions_out = [], []
    with httpx.Client(timeout=30) as client:
        for split in ("train", "validation", "test"):
            process_split(split, None, client, papers_out, questions_out)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "papers.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in papers_out) + "\n"
    )
    (OUT_DIR / "questions.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in questions_out) + "\n"
    )
    print(f"\nWrote {len(papers_out)} papers, {len(questions_out)} questions to {OUT_DIR}")


if __name__ == "__main__":
    main()
