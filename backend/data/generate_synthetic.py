"""
data/generate_synthetic.py - Fixed version
Run: python data/generate_synthetic.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Create output directory
OUT_DIR = Path(__file__).parent / "synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_tax_records(n=5000):
    print(f"Generating {n} tax records...")
    filing_status = np.random.choice(
        ["Single", "MarriedJointly", "MarriedSeparately", "HeadOfHousehold"],
        size=n, p=[0.45, 0.35, 0.10, 0.10]
    )
    income = np.random.lognormal(mean=11.0, sigma=0.8, size=n).round(2)
    investment_income = (np.random.exponential(scale=5000, size=n) *
                         np.random.binomial(1, 0.4, size=n)).round(2)
    business_income = (np.random.lognormal(mean=9.5, sigma=1.2, size=n) *
                       np.random.binomial(1, 0.25, size=n)).round(2)
    deductions = np.clip(income * np.random.uniform(0.05, 0.35, size=n), 2000, 80000).round(2)
    dependents = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.45, 0.25, 0.20, 0.08, 0.02])
    age = np.random.randint(22, 75, size=n)
    state = np.random.choice(["CA", "TX", "NY", "FL", "WA", "IL", "PA", "OH", "GA", "NC"], size=n)
    retirement = np.clip(np.random.normal(loc=8000, scale=4000, size=n), 0, 23000).round(2)

    taxable = np.maximum(income + investment_income + business_income - deductions - dependents * 4300, 0)

    def bracket_tax(ti):
        brackets = [(11600, 0.10), (47150, 0.12), (100525, 0.22),
                    (191950, 0.24), (243725, 0.32), (609350, 0.35)]
        tax, prev = 0.0, 0.0
        for limit, rate in brackets:
            if ti <= 0:
                break
            taxable_amt = min(ti, limit - prev)
            tax += taxable_amt * rate
            ti -= taxable_amt
            prev = limit
        if ti > 0:
            tax += ti * 0.37
        return round(tax, 2)

    tax_liability = np.array([bracket_tax(t) for t in taxable])
    effective_rate = np.where(income > 0, tax_liability / income, 0).round(4)
    anomaly_mask = np.random.binomial(1, 0.05, size=n).astype(bool)
    deductions[anomaly_mask] = (income[anomaly_mask] *
                                np.random.uniform(0.7, 1.5, size=anomaly_mask.sum())).round(2)

    df = pd.DataFrame({
        "record_id": [f"REC{str(i).zfill(5)}" for i in range(n)],
        "filing_status": filing_status,
        "age": age,
        "state": state,
        "gross_income": income,
        "investment_income": investment_income,
        "business_income": business_income,
        "total_deductions": deductions,
        "retirement_contributions": retirement,
        "dependents": dependents,
        "taxable_income": taxable.round(2),
        "tax_liability": tax_liability,
        "effective_tax_rate": effective_rate,
        "is_anomaly": anomaly_mask.astype(int),
    })

    out_path = OUT_DIR / "tax_records.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ tax_records.csv — {n} rows saved to {out_path}")
    return df


def generate_document_dataset(n=2000):
    print(f"Generating {n} documents...")
    NAMES = ["John Smith", "Jane Doe", "Michael Johnson", "Emily Davis"]
    COMPANIES = ["Acme Corp", "TechStart Inc", "Global Solutions LLC"]
    SERVICES = ["Software development", "Marketing consulting", "Legal services"]

    templates = {
        "W2": "Form W-2 Wage and Tax Statement. Employer: {co}. Wages: ${w:,.0f}. Federal tax withheld: ${t:,.0f}.",
        "1099": "Form 1099-MISC. Payer: {co}. Recipient: {n}. Nonemployee compensation: ${w:,.0f}.",
        "invoice": "INVOICE #{i}. Bill To: {n}. Services: {s}. Amount due: ${w:,.0f}.",
        "receipt": "Receipt #{i}. Merchant: {co}. Items: {s}. Total: ${w:,.0f}. Date: 2024.",
        "tax_return": "Form 1040 Tax Return. Taxpayer: {n}. Total income: ${w:,.0f}. Total tax: ${t:,.0f}.",
    }

    rows = []
    per_class = n // len(templates)
    for label, tmpl in templates.items():
        for _ in range(per_class):
            text = tmpl.format(
                co=random.choice(COMPANIES),
                n=random.choice(NAMES),
                s=random.choice(SERVICES),
                w=random.uniform(5000, 200000),
                t=random.uniform(500, 40000),
                i=random.randint(1000, 9999),
            )
            rows.append({"text": text, "document_type": label})

    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["doc_id"] = [f"DOC{str(i).zfill(5)}" for i in range(len(df))]

    out_path = OUT_DIR / "documents.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ documents.csv — {len(df)} rows saved to {out_path}")
    return df


def generate_qa_knowledge_base():
    print("Generating Q&A knowledge base...")
    qa_pairs = [
        {"q": "What is the standard deduction for 2024?",
         "a": "For 2024: $14,600 single, $29,200 married jointly, $21,900 head of household."},
        {"q": "What are the 2024 federal income tax brackets?",
         "a": "10% up to $11,600, 12% to $47,150, 22% to $100,525, 24% to $191,950, 32% to $243,725, 35% to $609,350, 37% above."},
        {"q": "What is the capital gains tax rate?",
         "a": "Long-term gains taxed at 0%, 15%, or 20% based on income. Short-term gains taxed as ordinary income."},
        {"q": "Can I deduct home office expenses?",
         "a": "Self-employed can deduct home office if used exclusively for business. Use simplified ($5/sqft) or regular method."},
        {"q": "What is the 401k contribution limit for 2024?",
         "a": "$23,000 for 2024, plus $7,500 catch-up if age 50+, totaling $30,500."},
        {"q": "What is a W-2 form?",
         "a": "Employer form showing annual wages and taxes withheld. Used to file federal and state returns."},
        {"q": "What is a 1099 form?",
         "a": "Reports non-wage income: 1099-NEC for contractors, 1099-INT for interest, 1099-DIV for dividends."},
        {"q": "What are estimated taxes?",
         "a": "Quarterly payments for self-employment/investment income. Due April 15, June 17, September 16, January 15."},
        {"q": "What expenses can a freelancer deduct?",
         "a": "Home office, vehicle, equipment, software, professional development, health insurance, half of SE tax."},
        {"q": "What is the child tax credit for 2024?",
         "a": "Up to $2,000 per child under 17, $1,600 refundable. Phases out at $200k single, $400k married."},
        {"q": "What triggers an IRS audit?",
         "a": "High deduction ratios, unreported income, large charitable deductions, repeated business losses."},
        {"q": "What is the self-employment tax rate?",
         "a": "15.3% on net SE income: 12.4% Social Security + 2.9% Medicare. Deduct half from AGI."},
        {"q": "What is a Schedule C?",
         "a": "Reports sole proprietor business profit/loss filed with Form 1040. Net profit subject to SE tax."},
        {"q": "What is a Roth IRA?",
         "a": "After-tax retirement account. Contributions not deductible but qualified withdrawals tax-free. 2024 limit $7,000."},
        {"q": "What is AMT?",
         "a": "Alternative Minimum Tax ensures minimum payment. 2024 exemption $85,700 single. Pay higher of regular or AMT."},
    ]

    out_path = OUT_DIR / "qa_knowledge_base.json"
    out_path.write_text(json.dumps(qa_pairs, indent=2))
    print(f"✅ qa_knowledge_base.json — {len(qa_pairs)} pairs saved to {out_path}")
    return qa_pairs


if __name__ == "__main__":
    print("🔧 Generating TaxMind synthetic datasets...\n")
    generate_tax_records(n=5000)
    generate_document_dataset(n=2000)
    generate_qa_knowledge_base()
    print("\n✅ All datasets saved to data/synthetic/")