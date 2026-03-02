"""
agent/taxmind_agent.py  (updated)
─────────────────────────────────────────────────────────────────────────────
TaxMind Hybrid Agent — now with multi-LLM routing support.

Pass `provider` to switch the brain of the agent at runtime:
    run_agent("...", provider="openai")
    run_agent("...", provider="claude")
    run_agent("...", provider="gemini")
    run_agent("...", provider="auto", task="long_document")
"""

from __future__ import annotations
import json
from typing import Any

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.llm_router import get_llm


# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def predict_tax_liability(
    gross_income: float,
    total_deductions: float,
    filing_status: str = "Single",
    investment_income: float = 0.0,
    business_income: float = 0.0,
    retirement_contributions: float = 0.0,
    dependents: int = 0,
    age: int = 35,
    state: str = "CA",
) -> str:
    """Predict a taxpayer's estimated federal tax liability and effective tax rate."""
    try:
        from models.tax_predictor import predict
        result = predict({
            "gross_income": gross_income, "total_deductions": total_deductions,
            "filing_status": filing_status, "investment_income": investment_income,
            "business_income": business_income,
            "retirement_contributions": retirement_contributions,
            "dependents": dependents, "age": age, "state": state,
        })
        return json.dumps({
            "predicted_tax_liability": f"${result['predicted_tax_liability']:,.2f}",
            "effective_tax_rate": result["effective_tax_rate_pct"],
            "net_after_tax": f"${max(gross_income - result['predicted_tax_liability'], 0):,.2f}",
        })
    except FileNotFoundError:
        return "❌ Tax predictor not trained. Run: python models/train_all.py"
    except Exception as e:
        return f"❌ Error: {str(e)}"


@tool
def classify_tax_document(document_text: str) -> str:
    """Classify a tax document from its text (W2, 1099, invoice, receipt, tax_return)."""
    try:
        from models.doc_classifier import classify
        result = classify(document_text)
        probs = ", ".join(f"{k}: {v:.0%}" for k, v in
                          sorted(result["probabilities"].items(), key=lambda x: -x[1]))
        return (f"Classified as: {result['predicted_class']} "
                f"(confidence: {result['confidence']:.0%}) | {probs}")
    except FileNotFoundError:
        return "❌ Document classifier not trained. Run: python models/train_all.py"
    except Exception as e:
        return f"❌ Error: {str(e)}"


@tool
def detect_tax_anomaly(
    gross_income: float, total_deductions: float, taxable_income: float,
    effective_tax_rate: float, tax_liability: float = 0.0,
    investment_income: float = 0.0, business_income: float = 0.0,
    filing_status: str = "Single", state: str = "CA",
    dependents: int = 0, retirement_contributions: float = 0.0,
) -> str:
    """Analyze a tax record for anomalies, fraud indicators, or audit risk."""
    try:
        from models.anomaly_detector import detect
        result = detect({
            "gross_income": gross_income, "total_deductions": total_deductions,
            "taxable_income": taxable_income, "effective_tax_rate": effective_tax_rate,
            "tax_liability": tax_liability, "investment_income": investment_income,
            "business_income": business_income, "filing_status": filing_status,
            "state": state, "dependents": dependents,
            "retirement_contributions": retirement_contributions,
        })
        return json.dumps({
            "anomaly_detected": result["is_anomaly"],
            "risk_level": result["risk_level"],
            "risk_score": f"{result['risk_score']:.0%}",
            "explanation": result["explanation"],
        })
    except FileNotFoundError:
        return "❌ Anomaly detector not trained. Run: python models/train_all.py"
    except Exception as e:
        return f"❌ Error: {str(e)}"


@tool
def search_tax_knowledge(query: str, top_k: int = 3) -> str:
    """Search the tax knowledge base for rules, deductions, brackets, and regulations."""
    try:
        from models.tax_qa import retrieve
        results = retrieve(query, top_k=top_k)
        if not results:
            return "No relevant tax information found."
        return "\n\n".join(
            f"[{i+1}] Q: {r['question']}\n    A: {r['answer']}"
            for i, r in enumerate(results)
        )
    except FileNotFoundError:
        return "❌ Q&A vector store not built. Run: python models/train_all.py"
    except Exception as e:
        return f"❌ Error: {str(e)}"


TOOLS = [predict_tax_liability, classify_tax_document, detect_tax_anomaly, search_tax_knowledge]

SYSTEM_PROMPT = """You are TaxMind, an expert AI tax assistant powered by machine learning.

You have four specialized ML tools:
1. predict_tax_liability — Estimate federal tax from income data
2. classify_tax_document — Identify document types (W2, 1099, invoice, etc.)
3. detect_tax_anomaly — Flag suspicious patterns and audit risk
4. search_tax_knowledge — Look up tax rules, deductions, and regulations

Always use tools when you have relevant numbers or document text.
Format dollar amounts clearly. Recommend consulting a licensed CPA for binding advice."""


# ─── Agent builder ────────────────────────────────────────────────────────────

def _build_agent(provider: str, task: str | None) -> AgentExecutor:
    llm = get_llm(provider=provider, task=task, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent, tools=TOOLS, verbose=True,
        max_iterations=6, return_intermediate_steps=True,
        handle_parsing_errors=True,
    )


# ─── Main entry point ─────────────────────────────────────────────────────────

def run_agent(
    message: str,
    history: list[dict] | None = None,
    provider: str = "openai",
    task: str | None = None,
) -> dict[str, Any]:
    """
    Run the TaxMind agent.

    Args:
        message:  User input.
        history:  [{"role": "user"|"assistant", "content": str}]
        provider: "openai" | "claude" | "gemini" | "gemini-lite" |
                  "claude-lite" | "openai-powerful" | "auto"
        task:     When provider="auto", routes to best model for the task.
    """
    agent_executor = _build_agent(provider=provider, task=task)

    chat_history = []
    for msg in (history or []):
        if msg.get("role") == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    result = agent_executor.invoke({"input": message, "chat_history": chat_history})

    tools_used = [
        {"tool": action.tool, "input": action.tool_input}
        for action, _ in result.get("intermediate_steps", [])
    ]

    return {
        "answer": result["output"],
        "provider_used": provider,
        "tools_used": tools_used,
        "tools_count": len(tools_used),
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from agent.llm_router import list_models
    print("🤖 TaxMind Agent — Interactive Mode")
    print("Available providers:", list(list_models().keys()))
    provider = input("Choose provider (default: openai): ").strip() or "openai"
    print(f"Using: {provider}\nType 'exit' to quit\n")

    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        try:
            result = run_agent(user_input, history=history, provider=provider)
            print(f"\nTaxMind [{provider}]: {result['answer']}")
            if result["tools_used"]:
                print(f"  [Tools: {', '.join(t['tool'] for t in result['tools_used'])}]")
            print()
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result["answer"]})
        except Exception as e:
            print(f"❌ Error: {e}\n")
