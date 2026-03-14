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
    domain:          str
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
    if isinstance(doc, dict):
        return doc.get("ratio_decidendi") or doc.get("answer") or doc.get("text") or doc.get("content") or str(doc)
    return str(doc)

def guardrail_node(state):
    result = _llm().invoke([HumanMessage(content=
        f"Is this a tax, financial, or accounting question? Answer YES or NO only.\nQuery: {state['query']}")])
    is_tax = "YES" in result.content.upper()
    print(f"Guardrail: {'Tax query' if is_tax else 'Non-tax query'}")
    return {**state, "is_tax_query": is_tax}

def query_rewriter_node(state):
    domain = state.get("domain", "it")
    domain_hint = "GST (Goods and Services Tax)" if domain == "gst" else "Income Tax / ITAT"
    result = _llm().invoke([HumanMessage(content=
        f"Rewrite this {domain_hint} question to be more specific for legal search. "
        f"Return only the rewritten question.\nOriginal: {state['query']}")])
    rewritten = result.content.strip()
    print(f"Rewritten: {rewritten}")
    return {**state, "rewritten_query": rewritten, "retry_count": state["retry_count"] + 1}

def retrieval_node(state):
    query = state.get("rewritten_query") or state["query"]
    domain = state.get("domain", "it")
    try:
        if domain == "gst":
            from services.qdrant_service import search_gst
            docs = search_gst(query, top_k=5)
            print(f"Retrieved {len(docs)} GST docs from Qdrant")
        else:
            from services.qdrant_service import search_judgments
            docs = search_judgments(query, top_k=5)
            print(f"Retrieved {len(docs)} IT judgment docs from Qdrant")
    except Exception as e:
        print(f"Qdrant retrieval failed: {e}")
        docs = []
    return {**state, "retrieved_docs": docs, "tools_used": state["tools_used"] + ["qdrant_search"]}

def grader_node(state):
    grades = []
    for doc in state["retrieved_docs"]:
        text = _doc_text(doc)[:1000]
        r = _llm().invoke([HumanMessage(content=
            f"Is this document relevant to the question? YES or NO only.\n"
            f"Q: {state['query']}\nDoc: {text}")])
        grades.append("YES" in r.content.upper())
    print(f"Grading: {sum(grades)}/{len(grades)} relevant")
    return {**state, "doc_grades": grades}

def generator_node(state):
    relevant = [d for d, g in zip(state["retrieved_docs"], state["doc_grades"]) if g]
    domain = state.get("domain", "it")

    if not relevant:
        context = "No relevant context found in the knowledge base."
    elif domain == "gst":
        context = "\n\n".join(
            f"[{i+1}] "
            f"Sections: {', '.join(d.get('sections', []))}\n"
            f"Content: {d.get('ratio_decidendi') or d.get('text', '')}\n"
            f"Circular: {d.get('circular_number', '')}\n"
            f"Source: {d.get('source_url', '')}"
            for i, d in enumerate(relevant)
        )
    else:
        context = "\n\n".join(
            f"[{i+1}] Court: {d.get('court', 'ITAT')} | "
            f"Outcome: {d.get('outcome', '')} | "
            f"Risk: {d.get('risk_level', '')}\n"
            f"Sections: {', '.join(d.get('sections', []))}\n"
            f"Trigger: {d.get('litigation_trigger', '')}\n"
            f"Ratio: {d.get('ratio_decidendi', '')}\n"
            f"Winning argument: {d.get('winning_argument', '')}\n"
            f"Mitigation: {'; '.join(d.get('mitigation_signals', []))}\n"
            f"Source: {d.get('source_url', '')}"
            for i, d in enumerate(relevant)
        )

    domain_context = "GST law and CBIC circulars" if domain == "gst" else "Income Tax Act and ITAT/High Court judgments"

    result = _llm().invoke([HumanMessage(content=
        f"You are TaxMind, an expert Indian tax assistant specializing in {domain_context}.\n\n"
        f"Use the following context to answer the question precisely.\n"
        f"For IT queries: cite judgment outcomes, ratio decidendi, and risk level.\n"
        f"For GST queries: cite relevant sections, circulars, and rates.\n"
        f"Always recommend consulting a licensed CA for specific advice.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}")])

    print(f"Answer generated ({len(result.content)} chars)")
    return {**state, "final_answer": result.content, "tools_used": state["tools_used"] + ["llm_generator"]}

def rejection_node(state):
    return {**state, "final_answer": (
        "I'm TaxMind, specialized in Indian tax questions. "
        "Please ask me about Income Tax, GST, deductions, filing, ITAT judgments, or related topics."
    )}

def route_guardrail(state):
    return "rewriter" if state["is_tax_query"] else "rejection"

def route_grader(state):
    if sum(state["doc_grades"]) >= 1:
        return "generator"
    if state["retry_count"] < 2:
        return "rewriter"
    return "generator"

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("guardrail", guardrail_node)
    g.add_node("rewriter",  query_rewriter_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("grader",    grader_node)
    g.add_node("generator", generator_node)
    g.add_node("rejection", rejection_node)
    g.set_entry_point("guardrail")
    g.add_conditional_edges("guardrail", route_guardrail, {
        "rewriter":  "rewriter",
        "rejection": "rejection"
    })
    g.add_edge("rewriter",  "retrieval")
    g.add_edge("retrieval", "grader")
    g.add_conditional_edges("grader", route_grader, {
        "rewriter":  "rewriter",
        "generator": "generator"
    })
    g.add_edge("generator", END)
    g.add_edge("rejection", END)
    return g.compile()

_graph = None

def run_langgraph_agent(query, provider=None, history=None, domain="it"):
    global _graph
    if provider:
        os.environ["DEFAULT_LLM_PROVIDER"] = provider

    cached = get_exact(query)
    if cached:
        return {**cached, "cached": True}
    cached = get_semantic(query)
    if cached:
        return {**cached, "cached": True}

    if _graph is None:
        _graph = build_graph()

    result = _graph.invoke(AgentState(
        query=query,
        domain=domain,
        rewritten_query=None,
        retrieved_docs=[],
        doc_grades=[],
        final_answer=None,
        is_tax_query=None,
        retry_count=0,
        tools_used=[],
        cached=False,
    ))

    output = {
        "answer":      result["final_answer"],
        "tools_used":  result["tools_used"],
        "retry_count": result["retry_count"],
        "cached":      False,
        "domain":      domain,
    }
    set_exact(query, output)
    set_semantic(query, output)
    return output