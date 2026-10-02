"""파일럿 실행·채점·사람 판정을 모아 최종 표를 낸다.

    python eval/analyze.py eval/runs/pilot-xxx.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import comb
from pathlib import Path

from review import kappa

METHODS = ("A", "B", "C", "D", "D-simple", "D-complex")
PAIRS = (("A", "B"), ("B", "C"), ("C", "D"), ("B", "D"), ("D", "D-complex"),
         ("D", "D-simple"), ("D-simple", "D-complex"), ("C", "D-complex"))
JUDGED = "judged-medium-v2"
HUMAN_FILES = ("human-r2", "human-fresh")  # 1회차는 기준 오해로 제외, 일치율 기록용
# 2회차 판정 뒤 확정한 기준(근거와 어긋나는 덧붙임은 오답)을 적용한다.
OVERRIDES = {("q27", "C"): "오답", ("q27", "D-complex"): "오답"}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verdicts(path: Path) -> dict[tuple[str, str], str]:
    return {(r["id"], r["method"]): r["verdict"] for r in load(path) if "verdict" in r}


def mcnemar(v: dict, ids: list[str], a: str, b: str) -> tuple[int, int, float]:
    """정확 이항 검정, 양측."""
    x = sum(v[(q, a)] == "정답" and v[(q, b)] != "정답" for q in ids)
    y = sum(v[(q, a)] != "정답" and v[(q, b)] == "정답" for q in ids)
    n = x + y
    if n == 0:
        return x, y, 1.0
    return x, y, min(1.0, 2 * sum(comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n)


def agreement(judge: dict, human: dict) -> str:
    pairs = [(judge[k], h) for k, h in human.items()]
    binary = [(a == "정답", b == "정답") for a, b in pairs]
    hit = sum(a == b for a, b in binary)
    return (f"{hit}/{len(binary)} ({hit / len(binary):.0%})  κ={kappa(binary):.2f}  "
            f"| 3단계 {sum(a == b for a, b in pairs)}/{len(pairs)} κ={kappa(pairs):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    def sibling(tag: str) -> Path:
        return args.run.with_name(f"{args.run.stem}.{tag}.jsonl")

    runs = {(r["id"], r["method"]): r for r in load(args.run)}
    judge = verdicts(sibling(JUDGED))
    human = {}
    for tag in HUMAN_FILES:
        human |= verdicts(sibling(tag))
    human |= OVERRIDES
    final = judge | human
    ids = sorted({q for q, _ in runs})

    print(f"판정 출처: 사람 {len(human)}건 + 채점 모델 {len(judge) - len(human)}건 "
          f"(사람 판정 덮어쓰기 {len(OVERRIDES)}건)\n")

    print("## 방식별 결과")
    print(f"{'방식':<10}{'정답(최종)':>9}{'정답(모델)':>9}{'오답(최종)':>9}{'비용':>9}  모델별 입력/출력 토큰")
    for m in METHODS:
        rows = [runs[(q, m)] for q in ids]
        tokens: Counter = Counter()
        for r in rows:
            for model, t in r["tokens"].items():
                tokens[(model.split("-")[1], "in")] += t["input"]
                tokens[(model.split("-")[1], "out")] += t["output"]
        by_model = ", ".join(f"{name} {tokens[(name, 'in')]:,}/{tokens[(name, 'out')]:,}"
                             for name in sorted({k[0] for k in tokens}))
        print(f"{m:<10}{sum(final[(q, m)] == '정답' for q in ids):>9}"
              f"{sum(judge[(q, m)] == '정답' for q in ids):>9}"
              f"{sum(final[(q, m)] == '오답' for q in ids):>9}"
              f"{sum(r['cost_usd'] for r in rows):>9.2f}  {by_model}")

    print("\n## McNemar (앞 방식만 정답 : 뒤 방식만 정답)")
    for a, b in PAIRS:
        fx, fy, fp = mcnemar(final, ids, a, b)
        jx, jy, jp = mcnemar(judge, ids, a, b)
        mark = "둘 다 유의" if fp < 0.05 and jp < 0.05 else ""
        print(f"{a:>9} vs {b:<10} 최종 {fx}:{fy} p={fp:.3f}   모델 {jx}:{jy} p={jp:.3f}   {mark}")

    print("\n## 파일럿 질문")
    d = [runs[(q, "D")] for q in ids]
    print(f"분류 판정 (D)        {dict(Counter(r['complexity'] for r in d))}")
    single = [r for r in runs.values() if r["method"] in ("D", "D-simple") and r["complexity"] == "simple"]
    print(f"인용검사 승급        {sum(r['escalated'] for r in single)}회 / 단순 경로 {len(single)}회")
    retry = [r for r in runs.values() if sum(s.startswith("검증") for s in r["trace"]) > 1]
    print(f"검증 후 재검색       {len(retry)}회 {sorted((r['id'], r['method']) for r in retry)}")
    print(f"지어낸 인용          {sum(bool(r['invented']) for r in runs.values())}건")
    print(f"키워드 폴백 섞인 실행 {sum(r.get('fallback_searches', 0) > 0 for r in runs.values())}건")
    c = [runs[(q, "C")] for q in ids]
    print(f"C 평균 검색 {sum(r['trace'].count('검색') for r in c) / len(c):.2f}회, "
          f"섹션 전문 조회 {sum('섹션조회' in r['trace'] for r in c)}/{len(c)}")

    print("\n## 채점 검증 (정답/비정답 일치, 기준 90%)")
    v1 = verdicts(sibling("judged-medium"))
    print(f"v1 × 1회차 40건 (공식)   {agreement(v1, verdicts(sibling('human')))}")
    print(f"v1 × 2회차 40건          {agreement(v1, verdicts(sibling('human-r2')))}")
    print(f"v2 × 2회차 40건 (참고)   {agreement(judge, verdicts(sibling('human-r2')))}")
    print(f"v2 × 새 표본 30건 (공식) {agreement(judge, verdicts(sibling('human-fresh')))}")


if __name__ == "__main__":
    main()
