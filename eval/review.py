"""채점 모델 판정을 사람이 블라인드로 재판정해 일치율을 낸다.

    python eval/review.py eval/runs/pilot-xxx.judged-medium.jsonl            # 판정 (이어서 가능)
    python eval/review.py eval/runs/pilot-xxx.judged-medium.jsonl --report   # 일치율만
    python eval/review.py eval/runs/pilot-xxx.judged-medium.jsonl --round 2  # 재판정, 결과는 별도 파일
    python eval/review.py eval/runs/pilot-xxx.judged-medium-v2.jsonl --fresh # 앞선 판정 건을 뺀 새 표본
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from judge import GROUND_TRUTH, reference

SAMPLE_SIZE = 40
FRESH_SIZE = 30  # 기준 수정 후 검증용, 앞선 판정 건 제외
SEED = 0
MIN_A = 4  # 회피·오답 경계 확인용
PASS_AGREEMENT = 0.90  # 정답/비정답 일치율. 채점 전에 확정, v2 검증에도 동일
SPLIT_METHODS = ("B", "C", "D", "D-simple", "D-complex")
KEYS = {"1": "정답", "2": "회피", "3": "오답"}


def select(records: list[dict], round_no: int = 1, size: int = SAMPLE_SIZE,
           exclude: frozenset[tuple[str, str]] = frozenset()) -> list[dict]:
    """판정이 갈린 질문은 전부 넣고 나머지는 무작위로 채운다."""
    rng = random.Random(SEED)
    records = [r for r in records if (r["id"], r["method"]) not in exclude]
    by_question: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_question[r["id"]].append(r)

    picked = []
    for rows in by_question.values():
        verdicts = {r.get("verdict") for r in rows if r["method"] in SPLIT_METHODS}
        if len(verdicts) > 1:
            picked += [r for r in rows if r["method"] in SPLIT_METHODS]

    rest = [r for r in records if r not in picked]
    rest_a = [r for r in rest if r["method"] == "A"]
    picked += rng.sample(rest_a, min(MIN_A, len(rest_a), max(size - len(picked), 0)))
    rest = [r for r in rest if r not in picked]
    picked += rng.sample(rest, min(len(rest), max(size - len(picked), 0)))

    # 같은 질문끼리 묶어 근거를 한 번만 읽게 한다
    order_rng = rng if round_no == 1 else random.Random(SEED + round_no)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in picked:
        groups[r["id"]].append(r)
    order = list(groups)
    order_rng.shuffle(order)
    ordered = []
    for qid in order:
        order_rng.shuffle(groups[qid])
        ordered += groups[qid]
    return ordered


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[k] * right[k] for k in left) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def report(records: list[dict], human: dict[tuple[str, str], dict]) -> None:
    judged = {(r["id"], r["method"]): r.get("verdict") for r in records}
    pairs = [(judged[key], h["verdict"]) for key, h in human.items()]
    if not pairs:
        print("판정한 건이 없다.")
        return
    binary = [(a == "정답", b == "정답") for a, b in pairs]
    agree = sum(a == b for a, b in binary) / len(binary)
    print(f"\n사람 판정 {len(pairs)}건")
    print(f"  정답/비정답 일치 {sum(a == b for a, b in binary)}/{len(binary)} ({agree:.0%})  "
          f"κ={kappa(binary):.2f}")
    print(f"  3단계 일치     {sum(a == b for a, b in pairs)}/{len(pairs)}  κ={kappa(pairs):.2f}")
    print(f"  기준 {PASS_AGREEMENT:.0%} → {'통과' if agree >= PASS_AGREEMENT else '미달: 기준 문구 수정 후 재채점'}")
    for (qid, method), h in human.items():
        if judged[(qid, method)] != h["verdict"]:
            print(f"  불일치 {qid} {method:<9} 모델 {judged[(qid, method)]} / 사람 {h['verdict']}"
                  + (f"  메모: {h['note']}" if h.get("note") else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("judged", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--round", type=int, default=1, help="2 이상이면 같은 표본을 순서만 바꿔 다시 판정")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.judged.read_text(encoding="utf-8").splitlines()]
    records = [r for r in records if "verdict" in r]
    cases = {c["id"]: c for c in (json.loads(line) for line in
                                  GROUND_TRUTH.read_text(encoding="utf-8").splitlines() if line.strip())}

    stem = args.judged.name.split(".judged")[0]
    exclude: frozenset[tuple[str, str]] = frozenset()
    if args.fresh:
        suffix = ".human-fresh.jsonl"
        exclude = frozenset((h["id"], h["method"])
                            for path in args.judged.parent.glob(f"{stem}.human.jsonl")
                            for h in map(json.loads, path.read_text(encoding="utf-8").splitlines()))
    else:
        suffix = ".human.jsonl" if args.round == 1 else f".human-r{args.round}.jsonl"
    out = args.judged.with_name(stem + suffix)
    human: dict[tuple[str, str], dict] = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            h = json.loads(line)
            human[(h["id"], h["method"])] = h

    if not args.report:
        sample = (select(records, size=FRESH_SIZE, exclude=exclude) if args.fresh
                  else select(records, args.round))
        queue = [r for r in sample if (r["id"], r["method"]) not in human]
        total = len(queue) + len(human)
        refs: dict[str, str] = {}
        with out.open("a", encoding="utf-8") as f:
            for r in queue:
                case = cases[r["id"]]
                refs.setdefault(r["id"], reference(case))
                print("\033[2J\033[H", end="")
                print(f"──────────────── [{len(human) + 1} / {total}] ────────────────")
                print(f"질문      {case['query']}")
                print(f"참고 문구  {', '.join(case['must_contain'])}\n")
                print(f"정답 근거\n{refs[r['id']]}\n")
                print(f"답변\n{r['answer']}\n")
                note = ""
                while True:
                    key = input("판정 → 1 정답 / 2 회피 / 3 오답 / m 메모 / q 종료: ").strip()
                    if key == "q":
                        report(records, human)
                        return
                    if key == "m":
                        note = input("메모: ").strip()
                        continue
                    if key in KEYS:
                        break
                h = {"id": r["id"], "method": r["method"], "verdict": KEYS[key], "note": note}
                human[(r["id"], r["method"])] = h
                f.write(json.dumps(h, ensure_ascii=False) + "\n")
                f.flush()

    report(records, human)


if __name__ == "__main__":
    main()
