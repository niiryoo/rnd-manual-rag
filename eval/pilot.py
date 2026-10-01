"""정답셋 질의를 비교 방식마다 실행해 jsonl 로 남긴다. 채점은 하지 않는다.

    python eval/pilot.py                       # 30개 x 6방식
    python eval/pilot.py --ids q01,q19,q23     # 일부만
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rnd_rag.agents.llm import HAIKU, SONNET
from rnd_rag.agents.runner import METHODS, run

GROUND_TRUTH = Path(__file__).with_name("ground_truth.jsonl")
RUNS_DIR = Path(__file__).with_name("runs")

PRICE_PER_MTOK = {HAIKU: (1.0, 5.0), SONNET: (2.0, 10.0)}  # USD 입력/출력, 2026-09 기준


def cost(tokens: dict[str, dict[str, int]]) -> float:
    total = 0.0
    for model, t in tokens.items():
        price_in, price_out = PRICE_PER_MTOK[model]
        total += (t["input"] * price_in + t["output"] * price_out) / 1_000_000
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", help="쉼표로 구분한 질의 id. 없으면 전부")
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()

    queries = [json.loads(line) for line in GROUND_TRUTH.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if args.ids:
        wanted = set(args.ids.split(","))
        queries = [q for q in queries if q["id"] in wanted]
    methods = args.methods.split(",")

    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"pilot-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    spent: dict[str, float] = defaultdict(float)
    errors: dict[str, int] = defaultdict(int)

    with out.open("w", encoding="utf-8") as f:
        for q in queries:
            for method in methods:
                record = {"id": q["id"], "type": q["type"], "method": method}
                start = time.perf_counter()
                try:
                    r = run(method, q["query"])
                    tokens = {m: {"input": t.input_tokens, "output": t.output_tokens,
                                  "calls": t.calls} for m, t in r.usage.by_model.items()}
                    record |= {
                        "answer": r.answer,
                        "citations": r.citations,
                        "retrieved": r.retrieved,
                        "invented": [c for c in r.citations if c not in r.retrieved],
                        "complexity": r.complexity,
                        "escalated": r.escalated,
                        "trace": r.trace,
                        "tokens": tokens,
                        "cost_usd": round(cost(tokens), 5),
                    }
                    spent[method] += record["cost_usd"]
                except Exception as e:  # 한 건 실패로 전체를 멈추지 않는다
                    record["error"] = f"{type(e).__name__}: {e}"
                    errors[method] += 1
                record["seconds"] = round(time.perf_counter() - start, 1)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                status = record.get("error") or " → ".join(record["trace"])
                print(f"{q['id']} {method:<9} {record['seconds']:>5}s  {status}")

    print(f"\n{out}")
    for method in methods:
        print(f"  {method:<9} ${spent[method]:.4f}  오류 {errors[method]}건")
    print(f"  합계      ${sum(spent.values()):.4f}")


if __name__ == "__main__":
    main()
