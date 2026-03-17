from langgraph.graph import StateGraph, END

from orchestration.state import LitReviewState
from memory.vector_memory import VectorMemory
from agents.supervisor import supervisor_agent
from agents.researcher import research_agent
from agents.memory_agent import memory_agent
from agents.analyst import analyst_agent
from agents.summarizer import summarizer_agent


def build_graph():
    vector_mem = VectorMemory()   # fresh per request — no cross-contamination
    graph      = StateGraph(LitReviewState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("research",   research_agent)
    graph.add_node("memory",     lambda s: memory_agent(s, vector_mem))
    graph.add_node("analysis",   lambda s: analyst_agent(s, vector_mem))
    graph.add_node("summarize",  summarizer_agent)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda s: s["next_step"],
        {
            "research":  "research",
            "analysis":  "analysis",
            "summarize": "summarize",
            "end":       END,
        },
    )

    graph.add_edge("research",  "memory")
    graph.add_edge("memory",    "analysis")
    graph.add_edge("analysis",  "supervisor")
    graph.add_edge("summarize", END)

    return graph.compile()