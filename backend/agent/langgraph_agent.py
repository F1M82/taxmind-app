"""agent/langgraph_agent.py – LangGraph agent with guardrails, rewriting, grading, caching"""
import os
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict, Optional, List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from agent.llm_router import get_llm
from services.cache_service import get_exact, set_exact, get_semantic, set_semantic

class AgentState(TypedDict):
    query:           str
    rewritten_query: Optional[str]
    retrieved_docs:  List[dict]
    doc_grades:      List[bool]
    final_answer:    Optional[str]
    is_tax_query:    Optional[bool]
    retry_count:     int
    tools_used:      List[str]
    cached:          bool

DEFAULT_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "claude")

def _llm(): return get_llm(DEFAULT_PROVIDER, temperature=0)

def _doc_text(doc):
    """Safely extract text from a doc that may be a dict or a plain string."""
    if isinstance(doc, dict):
        return doc.get('answer', doc.get('text', doc.get('content', str(doc))))
    return str(doc)

def guardrail_node(state):
    result = _llm().invoke([HumanMessage(content=
        f"Is this a tax, financial, or accounting question? Answer YES or NO only.\nQuery: {state['query']}")])
    is_tax = "YES" in result.content.upper()
    print(f"Guardrail: {'Tax query' if is_tax else 'Non-tax query'}")
    return {**state, "is_tax_query": is_tax}

def query_rewriter_node(state):
    result = _llm().invoke([HumanMessage(content=
        f"Rewrite this tax question to be more specific for search. Return only the rewritten question.\nOriginal: {state['query']}")])
    rewritten = result.content.strip()
    print(f"Rewritten: {rewritten}")
    return {**state, "rewritten_query": rewritten, "retry_count": state["retry_count"] + 1}

def retrieval_node(state):
    query = state.get("rewritten_query") or state["query"]
    try:
        from services.qdrant_service import hybrid_search
        docs = hybrid_search(query, top_k=3)
        print(f"Retrieved {len(docs)} docs from Qdrant")
    except Exception as e:
        print(f"Qdrant fallback: {e}")
        from models.tax_qa import retrieve
        docs = [{"question": r["question"], "answer": r["answer"], "score": r["similarity_score"]}
                for r in retrieve(query, top_k=3)]
    return {**state, "retrieved_docs": docs, "tools_used": state["tools_used"] + ["hybrid_search"]}

def grader_node(state):
    grades = []
    for doc in state["retrieved_docs"]:
        text = _doc_text(doc)
        r = _llm().invoke([HumanMessage(content=
            f"Is this document relevant to the question? YES or NO only.\nQ: {state['query']}\nDoc: {text}")])
        grades.append("YES" in r.content.upper())
    print(f"Grading: {sum(grades)}/{len(grades)} relevant")
    return {**state, "doc_grades": grades}

def generator_node(state):
    relevant = [d for d, g in zip(state["retrieved_docs"], state["doc_grades"]) if g]
    context = "\n\n".join(f"[{i+1}] {_doc_text(d)}" for i, d in enumerate(relevant))
    result = _llm().invoke([HumanMessage(content=
        f"You are TaxMind, an expert Indian tax assistant.\n\nContext:\n{context or 'No relevant context.'}\n\nQuestion: {state['query']}\n\nProvide a clear answer. Always recommend consulting a licensed CA for specific advice.")])
    print(f"Answer generated ({len(result.content)} chars)")
    return {**state, "final_answer": result.content, "tools_used": state["tools_used"] + ["llm_generator"]}

def rejection_node(state):
    return {**state, "final_answer": "I'm TaxMind, specialized in Indian tax and financial questions. Please ask me about taxes, deductions, filing, or related topics."}

def route_guardrail(state):  return "rewriter" if state["is_tax_query"] else "rejection"
def route_grader(state):
    if sum(state["doc_grades"]) >= 1: return "generator"
    if state["retry_count"] < 2:      return "rewriter"
    return "generator"

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("guardrail",  guardrail_node)
    g.add_node("rewriter",   query_rewriter_node)
    g.add_node("retrieval",  retrieval_node)
    g.add_node("grader",     grader_node)
    g.add_node("generator",  generator_node)
    g.add_node("rejection",  rejection_node)
    g.set_entry_point("guardrail")
    g.add_conditional_edges("guardrail", route_guardrail, {"rewriter": "rewriter", "rejection": "rejection"})
    g.add_edge("rewriter",  "retrieval")
    g.add_edge("retrieval", "grader")
    g.add_conditional_edges("grader", route_grader, {"rewriter": "rewriter", "generator": "generator"})
    g.add_edge("generator", END)
    g.add_edge("rejection", END)
    return g.compile()

_graph = None

def run_langgraph_agent(query, provider=None, history=None):
    global _graph
    if provider: os.environ["DEFAULT_LLM_PROVIDER"] = provider

    cached = get_exact(query)
    if cached: return {**cached, "cached": True}
    cached = get_semantic(query)
    if cached: return {**cached, "cached": True}

    if _graph is None: _graph = build_graph()

    result = _graph.invoke(AgentState(
        query=query, rewritten_query=None, retrieved_docs=[], doc_grades=[],
        final_answer=None, is_tax_query=None, retry_count=0, tools_used=[], cached=False,
    ))
    output = {"answer": result["final_answer"], "tools_used": result["tools_used"],
              "retry_count": result["retry_count"], "cached": False}
    set_exact(query, output)
    set_semantic(query, output)
    return output