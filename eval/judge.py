"""파일럿 답변을 매뉴얼 근거와 대조해 정답/회피/오답으로 판정한다.

    python eval/judge.py eval/runs/pilot-xxx.jsonl                 # Batch API, 50% 할인
    python eval/judge.py eval/runs/pilot-xxx.jsonl --direct        # 일반 호출, 바로 끝남
    python eval/judge.py eval/runs/pilot-xxx.jsonl --effort medium
    python eval/judge.py eval/runs/pilot-xxx.jsonl --batch-id msgbatch_...   # 제출한 배치 이어받기
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import anthropic
from anthropic.types.messages.batch_create_params import Request

from rnd_rag.agents import nodes
from rnd_rag.agents.llm import OPUS, client
from rnd_rag.search import SearchResult

GROUND_TRUTH = Path(__file__).with_name("ground_truth.jsonl")
VERDICTS = ("정답", "회피", "오답")
PROMPT_VERSION = 2  # 결과 파일명에 붙음. 기준 수정은 v2 한 번만
PRICE_PER_MTOK = (4.0, 20.0)  # Opus 5.5 입력/출력 USD. 배치는 절반
POLL_SECONDS = 30
DIRECT_WORKERS = 6

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "description": "어느 사실이 근거와 맞거나 다른지 한 줄"},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
    },
    "required": ["reason", "verdict"],
    "additionalProperties": False,
}

PROMPT = """아래 매뉴얼 근거를 기준으로 답변을 판정해라.

정답 - 질문이 요구한 사항(금액·기간·비율·조건)을 근거와 일치하게 모두 답했다.
       표현이 달라도 뜻이 같으면 일치로 본다. 예: "100분의 150" = "150%"
오답 - 질문이 물은 사항에 대해 근거와 어긋나는 말을 했다.
       핵심을 맞혔더라도 같은 사항에 대해 근거와 어긋나는 말을 덧붙였으면 오답이다.
회피 - 근거와 어긋나는 말은 없지만, 질문이 요구한 사항을 다 제시하지 않았다.
       모른다고 답한 경우, 근거에 없다고 답한 경우, 일반론만 말한 경우,
       둘 중 하나만 답한 경우가 여기 해당한다.

판정 순서: 근거와 어긋나는 내용이 있으면 오답이다. 없으면 빠진 것이 있는지 본다.
근거로 확인할 수 없는 추가 설명은 판정에 쓰지 않는다.
참고 문구는 근거에서 정답 위치를 찾는 단서다. 질문이 넓으면 참고 문구의 내용을
빠짐없이 말하지 않았더라도, 질문이 요구한 범위를 근거대로 답했으면 정답이다.

# 질문
{query}

# 참고 문구
{facts}

# 매뉴얼 근거
{reference}

# 답변
{answer}"""


def reference(case: dict) -> str:
    """정답 섹션마다 must_contain 이 가장 많이 든 청크 주변을 발췌한다."""
    repo = nodes.service().repo
    results = []
    for section_id in case["answer_sections"]:
        section = repo.section(section_id)
        chunks = tuple(repo.section_chunks(section_id))
        anchor = max(chunks, key=lambda c: sum(s in c.text for s in case["must_contain"])).chunk_id
        results.append(SearchResult(section, chunks, 0.0, anchor))
    return nodes.evidence(tuple(results))


def build_params(records: list[dict], cases: dict[str, dict], effort: str) -> list[dict[str, Any]]:
    refs = {qid: reference(case) for qid, case in cases.items()}
    params = []
    for r in records:
        case = cases[r["id"]]
        prompt = PROMPT.format(query=case["query"], facts=", ".join(case["must_contain"]),
                               reference=refs[r["id"]], answer=r["answer"])
        params.append({
            "model": OPUS,
            "max_tokens": 8000,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": effort,
                              "format": {"type": "json_schema", "schema": SCHEMA}},
        })
    return params


def run_direct(params: list[dict[str, Any]]) -> list[anthropic.types.Message | str]:
    def one(p: dict[str, Any]) -> anthropic.types.Message | str:
        try:
            return client().messages.create(**p)
        except anthropic.APIError as e:
            return f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(DIRECT_WORKERS) as pool:
        return list(pool.map(one, params))


def run_batch(params: list[dict[str, Any]], batch_id: str | None) -> list[anthropic.types.Message | str]:
    if batch_id is None:
        requests = [Request(custom_id=f"r{i}", params=p) for i, p in enumerate(params)]
        batch_id = client().messages.batches.create(requests=requests).id
    print(f"배치 {batch_id}", flush=True)

    while (batch := client().messages.batches.retrieve(batch_id)).processing_status != "ended":
        print(f"  처리중 {batch.request_counts.processing}건", flush=True)
        time.sleep(POLL_SECONDS)

    outcomes: list[anthropic.types.Message | str] = ["missing"] * len(params)
    for result in client().messages.batches.results(batch_id):
        i = int(result.custom_id[1:])
        ok = result.result.type == "succeeded"
        outcomes[i] = result.result.message if ok else result.result.type
    return outcomes


def apply(records: list[dict], outcomes: list[anthropic.types.Message | str],
          discount: float) -> float:
    price_in, price_out = (p * discount for p in PRICE_PER_MTOK)
    spent = 0.0
    for record, msg in zip(records, outcomes):
        if isinstance(msg, str):
            record["judge_error"] = msg
            continue
        usage = msg.usage
        record["judge_tokens"] = {"input": usage.input_tokens, "output": usage.output_tokens}
        spent += (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1e6
        if msg.stop_reason != "end_turn":
            record["judge_error"] = msg.stop_reason
            continue
        data = json.loads("".join(b.text for b in msg.content if b.type == "text"))
        record["verdict"], record["reason"] = data["verdict"], data["reason"]
    return spent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--batch-id")
    args = parser.parse_args()

    lines = GROUND_TRUTH.read_text(encoding="utf-8").splitlines()
    cases = {c["id"]: c for c in (json.loads(line) for line in lines if line.strip())}
    records = [json.loads(line) for line in args.run.read_text(encoding="utf-8").splitlines()]
    records = [r for r in records if "error" not in r]  # 실행 실패 건 제외

    params = build_params(records, cases, args.effort)
    mode = "direct" if args.direct else "batch"
    print(f"{len(records)}건  effort={args.effort}  {mode}", flush=True)
    if args.direct:
        spent = apply(records, run_direct(params), discount=1.0)
    else:
        spent = apply(records, run_batch(params, args.batch_id), discount=0.5)

    out = args.run.with_name(f"{args.run.stem}.judged-{args.effort}-v{PROMPT_VERSION}.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                   encoding="utf-8")

    by_method: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        by_method[r["method"]][r.get("verdict", "채점실패")] += 1
    print(f"\n{out}")
    for method, counts in by_method.items():
        print(f"  {method:<9} " + "  ".join(f"{v} {counts[v]}" for v in (*VERDICTS, "채점실패")))
    print(f"  채점 비용 ${spent:.4f}")


if __name__ == "__main__":
    main()
