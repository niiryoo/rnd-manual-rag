"""질문 하나를 비교 방식 하나로 실행한다."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from rnd_rag.agents import graph, nodes
from rnd_rag.agents.llm import SONNET, call_json, call_tools
from rnd_rag.agents.nodes import ANSWER_TOKENS
from rnd_rag.agents.state import AgentState, Complexity, Usage
from rnd_rag.mcp.formatting import format_section
from rnd_rag.search import SearchResult

Method = Literal["A", "B", "C", "D", "D-simple", "D-complex"]
METHODS: tuple[Method, ...] = ("A", "B", "C", "D", "D-simple", "D-complex")

MAX_TOOL_ROUNDS = 6  # C 도구 호출 왕복 상한


@dataclass(frozen=True)
class RunResult:
    answer: str
    citations: tuple[str, ...]
    retrieved: tuple[str, ...]  # 모델이 본 section_id
    usage: Usage
    trace: tuple[str, ...]
    complexity: Complexity | None = None  # 분류기 판정 또는 강제한 값
    escalated: bool = False


def run(method: Method, query: str) -> RunResult:
    if method == "A":
        return _no_tools(query)
    if method == "B":
        return _search_once(query)
    if method == "C":
        return _single_agent(query)
    forced: dict[str, Complexity | None] = {"D": None, "D-simple": "simple",
                                            "D-complex": "complex"}
    return _from_state(graph.run(query, forced[method]))


def _section_ids(results: tuple[SearchResult, ...]) -> tuple[str, ...]:
    return tuple(r.section.section_id for r in results)


def _citations(raw: list[str]) -> tuple[str, ...]:
    return tuple(c.strip("[] ") for c in raw)


NO_TOOLS_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

NO_TOOLS_PROMPT = """정부 R&D 매뉴얼(국가연구개발혁신법 매뉴얼 본권과 별권 4종)에 관한 질문이다.
알고 있는 내용으로 답해라. 확실하지 않으면 그렇다고 말해라.
금액·기간·비율은 정확히 써라.

질문: {query}"""


def _no_tools(query: str) -> RunResult:
    data, usage = call_json(NO_TOOLS_PROMPT.format(query=query), NO_TOOLS_SCHEMA,
                            model=SONNET, max_tokens=ANSWER_TOKENS)
    return RunResult(data["answer"], (), (), usage, ("답변",))


def _search_once(query: str) -> RunResult:
    state: AgentState = {"query": query, "complexity": "simple"}
    found = nodes.retrieve(state)
    done = nodes.synthesize({**state, "retrieved": found["retrieved"]})
    return RunResult(done["answer"], done["citations"], _section_ids(found["retrieved"]),
                     done["usage"], found["trace"] + done["trace"])


TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_manual",
        "description": (
            "국가연구개발사업 규정을 매뉴얼에서 검색한다. 결과마다 [section_id] 와 출처 "
            "쪽번호가 붙는다. '(섹션 일부)' 로 표시된 결과는 get_section 으로 전문을 볼 수 있다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "검색어"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_section",
        "description": (
            "섹션 전문을 가져온다. 긴 섹션은 나뉘어 오며 끝에 표시된 offset 으로 이어서 본다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "section_id": {"type": "string"},
                "offset": {"type": "integer"},
            },
            "required": ["section_id"],
        },
    },
]

AGENT_PROMPT = """정부 R&D 매뉴얼 질문에 답해라. 도구로 매뉴얼을 찾아 확인한 내용만 근거로 쓴다.

근거에 없는 내용은 쓰지 마라. 찾지 못하면 그렇다고 말해라.
금액·기간·비율은 매뉴얼에 적힌 표현 그대로 옮겨라.

질문: {query}"""


def _run_tool(name: str, args: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    if name == "search_manual":
        results = tuple(nodes.service().search(args["query"], limit=nodes.SIMPLE_LIMIT).results)
        return nodes.evidence(results), _section_ids(results)

    repo = nodes.service().repo
    section = repo.section(args["section_id"])
    if section is None:
        return f"'{args['section_id']}' 섹션이 없다.", ()
    body = format_section(section.level_path, section.doc_id, section.citation,
                          repo.section_chunks(section.section_id), args.get("offset", 0))
    return body, (section.section_id,)


def _single_agent(query: str) -> RunResult:
    messages: list[dict[str, Any]] = [{"role": "user", "content": AGENT_PROMPT.format(query=query)}]
    seen: dict[str, None] = {}  # 본 순서 유지
    usage = Usage()
    trace: list[str] = []

    for round_no in range(MAX_TOOL_ROUNDS + 1):
        response, used = call_tools(messages, TOOLS, nodes.ANSWER_SCHEMA, model=SONNET,
                                    max_tokens=ANSWER_TOKENS,
                                    allow_tools=round_no < MAX_TOOL_ROUNDS)
        usage += used
        calls = [b for b in response.content if b.type == "tool_use"]
        if not calls:
            break
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for call in calls:
            text, ids = _run_tool(call.name, call.input)
            seen.update(dict.fromkeys(ids))
            trace.append("검색" if call.name == "search_manual" else "섹션조회")
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": text})
        messages.append({"role": "user", "content": results})

    text = "".join(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    return RunResult(data["answer"], _citations(data["citations"]), tuple(seen), usage,
                     (*trace, "답변"))


def _from_state(state: AgentState) -> RunResult:
    return RunResult(
        answer=state.get("answer", ""),
        citations=state.get("citations", ()),
        retrieved=_section_ids(state.get("retrieved", ())),
        usage=state.get("usage", Usage()),
        trace=state.get("trace", ()),
        complexity=state.get("complexity"),
        escalated=bool(state.get("escalated")),
    )
