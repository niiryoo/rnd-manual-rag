"""그래프의 노드. 각 노드는 상태를 읽고 바뀐 부분만 돌려준다."""

from __future__ import annotations

from functools import lru_cache

from rnd_rag.agents.llm import HAIKU, SONNET, call_json
from rnd_rag.agents.state import MAX_ATTEMPTS, AgentState
from rnd_rag.parsing.profiles import PROFILES
from rnd_rag.search import SearchResult, SearchService
from rnd_rag.store import Repository, open_db

SIMPLE_LIMIT = 3  # 정답셋에서 상위3 안에 93% 가 들어온다
COMPLEX_LIMIT = 3  # 하위 질문마다 가져오므로 늘리면 컨텍스트가 샌다
MAX_SUBQUERIES = 4

SNIPPET_CHARS = 2500  # 정답의 96% 가 이 안에 들어온다


@lru_cache(maxsize=1)
def service() -> SearchService:
    # LangGraph 가 노드를 워커 스레드에서 돌린다
    return SearchService(Repository(open_db(readonly=True, shared_threads=True)))


def _doc_title(doc_id: str) -> str:
    profile = PROFILES.get(doc_id)
    return profile.short_title if profile else doc_id


def _window(result: SearchResult, limit: int) -> tuple[str, bool]:
    """앞에서부터 자르면 뒤쪽 표가 통째로 날아간다."""
    chunks = result.chunks
    if not chunks:
        return "", False
    ids = [c.chunk_id for c in chunks]
    lo = hi = ids.index(result.matched_chunk_id) if result.matched_chunk_id in ids else 0
    used = len(chunks[lo].text)
    while True:
        grew = False
        if hi + 1 < len(chunks) and used + len(chunks[hi + 1].text) <= limit:
            hi += 1
            used += len(chunks[hi].text)
            grew = True
        if lo > 0 and used + len(chunks[lo - 1].text) <= limit:
            lo -= 1
            used += len(chunks[lo].text)
            grew = True
        if not grew:
            break
    body = "\n\n".join(c.text for c in chunks[lo : hi + 1])
    return body, (lo > 0 or hi + 1 < len(chunks))


def evidence(results: tuple[SearchResult, ...]) -> str:
    if not results:
        return "(검색 결과 없음)"
    blocks = []
    for result in results:
        section = result.section
        body, partial = _window(result, SNIPPET_CHARS)
        head = (f"[{section.section_id}] {' > '.join(section.level_path)}\n"
                f"출처: {_doc_title(section.doc_id)} {result.citation}")
        if partial:
            head += " (섹션 일부)"
        blocks.append(f"{head}\n\n{body}")
    return "\n\n---\n\n".join(blocks)


CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "complexity": {
            "type": "string",
            "enum": ["simple", "complex"],
            "description": "여러 항목을 엮어야 하면 complex, 아니면 simple",
        }
    },
    "required": ["complexity"],
}

CLASSIFY_PROMPT = """정부 R&D 매뉴얼 질문이다. 질문이 요구하는 형태만 보고 판정해라.
매뉴얼 내용을 추측하지 마라.

simple - 값 하나, 기준 하나, 절차 하나를 묻는다
  "간접비 비율 얼마야?"  "연구개발비 이월 기준"  "협약은 언제까지 체결해?"

complex - 답을 만들려면 여러 항목을 엮어야 한다. 아래 중 하나에 해당할 때만이다
  두 가지를 한꺼번에 물음  "반납이랑 이월이랑 둘 다 기한이 언제야?"
  준 숫자를 구간표에 대입  "간접비가 3천만원이면 한도 안에 드나?"
  두 대상을 비교          "학생연구자랑 참여연구자랑 보험 기준이 다른가?"
  둘 중 어느 쪽인지 물음   "현금으로 내야 하나 현물도 되나?"

위 네 가지에 해당하지 않으면 simple 이다. 애매하면 simple 이다.

질문: {query}"""


def classify(state: AgentState) -> AgentState:
    data, usage = call_json(
        CLASSIFY_PROMPT.format(query=state["query"]),
        CLASSIFY_SCHEMA, model=HAIKU, max_tokens=64,
    )
    return {"complexity": data["complexity"], "usage": usage, "trace": ("분류",)}


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "subqueries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "검색어. 매뉴얼 본문에 쓰일 법한 용어로 쓴다",
        }
    },
    "required": ["subqueries"],
}

PLAN_PROMPT = """정부 R&D 매뉴얼에서 답을 찾아야 한다. 이 질문을 검색어 2~{max}개로 쪼개라.

질문자의 표현이 아니라 매뉴얼에 실제로 쓰일 법한 용어로 바꿔 써라.
예: "로열티" -> "기술료율", "기관부담금" -> "기관부담연구개발비"

질문: {query}"""


def plan(state: AgentState) -> AgentState:
    data, usage = call_json(
        PLAN_PROMPT.format(query=state["query"], max=MAX_SUBQUERIES),
        PLAN_SCHEMA, model=HAIKU, max_tokens=400,
    )
    subqueries = tuple(data["subqueries"][:MAX_SUBQUERIES]) or (state["query"],)
    return {"subqueries": subqueries, "usage": usage, "trace": ("계획",)}


def _is_complex(state: AgentState) -> bool:
    return state.get("complexity") == "complex" or bool(state.get("escalated"))


def retrieve(state: AgentState) -> AgentState:
    gap = state.get("gap")
    # 하위 질문은 짧아서 원 질의보다 표를 덜 집는다
    queries = (gap,) if gap else (state["query"], *state.get("subqueries", ()))
    limit = COMPLEX_LIMIT if _is_complex(state) else SIMPLE_LIMIT

    per_query = [service().search(query, limit=limit).results for query in queries]
    # 한 질문이 6칸을 다 먹으면 "둘 다" 형 질문이 반쪽만 답한다
    merged: list[SearchResult] = []
    for rank in range(limit):
        merged += [results[rank] for results in per_query if rank < len(results)]
    # gap 을 비워야 다음 재검색이 새 지적을 쓴다
    return {"retrieved": tuple(merged), "gap": "", "trace": ("검색",)}


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "근거에 있는 내용만으로 답한다"},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "근거로 쓴 section_id. 대괄호 없이 값만. 예: v2-s013",
        },
    },
    "required": ["answer", "citations"],
}

ANSWER_PROMPT = """아래 매뉴얼 발췌만 근거로 질문에 답해라.

근거에 없는 내용은 쓰지 마라. 발췌로 답할 수 없으면 그렇다고 말해라.
금액·기간·비율은 발췌에 적힌 표현 그대로 옮겨라.

# 질문
{query}

# 매뉴얼 발췌
{evidence}"""


def _answer(state: AgentState, model: str, label: str) -> AgentState:
    data, usage = call_json(
        ANSWER_PROMPT.format(query=state["query"], evidence=evidence(state["retrieved"])),
        ANSWER_SCHEMA, model=model, max_tokens=2000,
    )
    return {
        "answer": data["answer"],
        # Haiku 는 [v2-s013] 로 돌려준다
        "citations": tuple(c.strip("[] ") for c in data["citations"]),
        "usage": usage,
        "trace": (label,),
    }


def answer(state: AgentState) -> AgentState:
    return _answer(state, HAIKU, "답변")


def synthesize(state: AgentState) -> AgentState:
    return _answer(state, SONNET, "종합")


VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["ok", "insufficient"]},
        "gap": {
            "type": "string",
            "description": "insufficient 면 더 찾아야 할 검색어 하나, ok 면 빈 문자열",
        },
    },
    "required": ["verdict", "gap"],
}

VERIFY_PROMPT = """답변이 근거만으로 성립하는지 검사해라. 답을 고쳐 쓰지는 마라.

insufficient 로 판정할 경우: 근거에 없는 수치나 조건을 답이 말하고 있을 때,
또는 질문이 요구한 항목 중 근거에 없는 것이 있을 때.

# 질문
{query}

# 답변
{answer}

# 근거
{evidence}"""


def invented_citations(state: AgentState) -> list[str]:
    """검색에 없던 section_id 를 인용했다면 지어낸 것이다."""
    known = {r.section.section_id for r in state.get("retrieved", ())}
    return [c for c in state.get("citations", ()) if c not in known]


def check_citations(state: AgentState) -> AgentState:
    """인용한 section_id 가 검색 결과에 있는지만 본다."""
    if not invented_citations(state):
        return {"verdict": "ok", "trace": ("인용검사",)}
    # 지어낸 인용은 근거가 모자랐다는 신호이기도 하다
    return {"verdict": "insufficient", "escalated": True, "trace": ("인용검사(승급)",)}


def verify(state: AgentState) -> AgentState:
    attempts = state.get("attempts", 0) + 1
    if invented_citations(state):
        return {"verdict": "insufficient", "gap": state["query"],
                "attempts": attempts, "trace": ("검증(인용오류)",)}

    data, usage = call_json(
        VERIFY_PROMPT.format(
            query=state["query"],
            answer=state.get("answer", ""),
            evidence=evidence(state["retrieved"]),
        ),
        VERIFY_SCHEMA, model=HAIKU, max_tokens=300,
    )
    return {
        "verdict": data["verdict"],
        "gap": data["gap"] if data["verdict"] == "insufficient" else "",
        "attempts": attempts,
        "usage": usage,
        "trace": ("검증",),
    }


def route_start(state: AgentState) -> str:
    # 판정을 받고 시작하면 분류를 건너뛴다
    return route_complexity(state) if "complexity" in state else "classify"


def route_complexity(state: AgentState) -> str:
    return "plan" if state.get("complexity") == "complex" else "retrieve"


def route_after_retrieve(state: AgentState) -> str:
    return "synthesize" if _is_complex(state) else "answer"


def route_citations(state: AgentState) -> str:
    return "end" if state.get("verdict") == "ok" else "synthesize"


def route_verdict(state: AgentState) -> str:
    if state.get("verdict") == "ok" or state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "end"
    return "retrieve"
