"""노드를 엣지로 묶는다. 실행 순서는 이 파일에만 있다."""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from rnd_rag.agents import nodes
from rnd_rag.agents.state import AgentState, Complexity


@lru_cache(maxsize=1)
def build():
    graph = StateGraph(AgentState)
    graph.add_node("classify", nodes.classify)
    graph.add_node("plan", nodes.plan)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("answer", nodes.answer)
    graph.add_node("check", nodes.check_citations)
    graph.add_node("synthesize", nodes.synthesize)
    graph.add_node("verify", nodes.verify)

    graph.add_conditional_edges(START, nodes.route_start,
                                {"classify": "classify", "plan": "plan", "retrieve": "retrieve"})
    graph.add_conditional_edges("classify", nodes.route_complexity,
                                {"plan": "plan", "retrieve": "retrieve"})
    graph.add_edge("plan", "retrieve")
    graph.add_conditional_edges("retrieve", nodes.route_after_retrieve,
                                {"answer": "answer", "synthesize": "synthesize"})
    graph.add_edge("answer", "check")
    # 단순 경로가 인용 검사에 걸리면 복합 경로로 올라간다
    graph.add_conditional_edges("check", nodes.route_citations,
                                {"end": END, "synthesize": "synthesize"})
    graph.add_edge("synthesize", "verify")
    graph.add_conditional_edges("verify", nodes.route_verdict,
                                {"end": END, "retrieve": "retrieve"})
    return graph.compile()


def run(query: str, complexity: Complexity | None = None) -> AgentState:
    state: AgentState = {"query": query}
    if complexity:
        state["complexity"] = complexity
    return build().invoke(state)
