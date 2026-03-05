import json, hashlib, os, argparse
from pathlib import Path
from datetime import datetime

for d in ["review_queue", "approved", "rejected", "logs"]:
    Path(d).mkdir(exist_ok=True)

OPENSEARCH_URL   = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "taxmind-kb")


def doc_id(t):
    return hashlib.md5(t.encode()).hexdigest()[:12]


def save_chunks(chunks, source):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("review_queue") / f"{source}_{ts}.json"
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved {len(chunks)} chunks -> {out}")


def make_chunk(text, source, meta):
    return {
        "id": doc_id(text),
        "text": text.strip(),
        "source": source,
        "metadata": {**meta, "extracted_at": datetime.now().isoformat()},
        "status": "pending"
    }


# ── Extractors ────────────────────────────────────────────────────────────────

def extract_it_act():
    print("\nExtracting Income Tax Act...")
    data = [
        ("10",      "Exemptions - Section 10 covers agricultural income, share of profit from firm, HRA, LTA, gratuity up to limits, leave encashment, scholarships, and other specified exemptions from total income."),
        ("80C",     "Deduction up to Rs 1.5 lakh for LIC premium, PPF, ELSS, NSC, home loan principal repayment, tuition fees for children, Sukanya Samriddhi, 5-year FD. Only under old tax regime."),
        ("80D",     "Medical insurance deduction: Rs 25000 for self and family, Rs 50000 if senior citizen. Additional Rs 25000 or Rs 50000 for parents. Preventive health checkup Rs 5000 within overall limit."),
        ("87A",     "Rebate under old regime Rs 12500 if total income up to Rs 5 lakh. Under new regime Rs 25000 rebate if total income up to Rs 7 lakh. Rebate not available on special rate income like LTCG."),
        ("115BAC",  "New Tax Regime default from AY 2024-25. Slabs: Nil up to 3L, 5% from 3-7L, 10% from 7-10L, 15% from 10-12L, 20% from 12-15L, 30% above 15L. Standard deduction Rs 75000. No 80C or 80D deductions."),
        ("44AD",    "Presumptive taxation for eligible business: 8 percent of turnover or 6 percent for digital receipts deemed as profit. Turnover limit Rs 3 crore if 95 percent receipts are digital. No books of accounts required."),
        ("44ADA",   "Presumptive taxation for professionals including doctors, lawyers, CAs, engineers, architects: 50 percent of gross receipts deemed as profit. Gross receipts limit Rs 75 lakh or Rs 3.75 lakh if 95 percent digital."),
        ("194A",    "TDS on interest other than securities at 10 percent. Threshold Rs 40000 per year per bank, Rs 50000 for senior citizens. Applicable on FD interest, recurring deposit interest, interest from NBFC."),
        ("194C",    "TDS on payments to contractors: 1 percent for individual or HUF, 2 percent for others. Threshold Rs 30000 per contract or Rs 100000 aggregate per year. Applies to work contracts, transport, catering."),
        ("234B",    "Interest for default in advance tax at 1 percent per month if advance tax paid is less than 90 percent of assessed tax. Computed from April 1 of assessment year to date of assessment or actual payment."),
        ("234C",    "Interest for deferment of advance tax instalments at 1 percent per month for 3 months on shortfall. Due dates: 15 June 15 percent, 15 September 45 percent, 15 December 75 percent, 15 March 100 percent."),
        ("148",     "Notice for income escaping assessment within 3 years if escaped income up to Rs 50 lakh, or 10 years if above Rs 50 lakh with evidence. Prior approval of specified authority required before issue."),
        ("43B_h",   "Section 43B(h): Payment to micro and small enterprises must be within 45 days with agreement or 15 days without agreement as per MSMED Act. Unpaid amount disallowed until actual payment. Effective AY 2024-25."),
        ("139_8A",  "Updated return under section 139(8A) can be filed within 2 years of end of assessment year. Additional tax 25 percent if within 12 months, 50 percent if between 12 to 24 months. Not for loss returns."),
    ]
    chunks = [make_chunk(f"Section {s} Income Tax Act 1961: {c}", "it_act",
              {"section_number": s, "act": "Income Tax Act 1961", "content_type": "statute", "language": "en"})
              for s, c in data]
    for s, _ in data:
        print(f"  -> Section {s}")
    save_chunks(chunks, "it_act")
    return chunks


def extract_cbdt():
    print("\nExtracting CBDT Circulars...")
    data = [
        ("Circular 1/2024",      "2024-01-01", "Extension of due date for Form 10-IC and Form 10-ID for AY 2021-22 and AY 2022-23 for companies opting for concessional tax rate under sections 115BAA and 115BAB of Income Tax Act."),
        ("Circular 2/2024",      "2024-03-15", "TDS on salary under section 192: Employer must ask employee to declare intended tax regime. Default is new regime. Employee can opt for old regime by submitting declaration. Form 12BAA for declaring other TDS and TCS."),
        ("Circular 9/2023",      "2023-09-28", "Condonation of delay in filing Form 10-IC for AY 2020-21. Domestic companies opting for 22 percent concessional rate under section 115BAA can file with condonation application to PCIT or CIT."),
        ("Notification 35/2023", "2023-06-29", "Form 12BAA notified for employees to inform employer about TDS and TCS deducted by other deductors and other income sources. Enables correct TDS computation on salary by employer under section 192."),
        ("Circular 4/2023",      "2023-04-05", "Guidelines for compulsory scrutiny selection for FY 2023-24. Cases with foreign assets, high value cash transactions, survey cases, and specific high-risk parameters to be selected for complete scrutiny."),
        ("Circular 12/2022",     "2022-06-16", "Vivad Se Vishwas Scheme extension. Taxpayers who filed declaration but did not pay within time granted extension. Late payment attracts interest at 1.5 times normal rate on outstanding amount."),
    ]
    chunks = [make_chunk(f"CBDT {num} dated {date}: {content}", "cbdt",
              {"circular_number": num, "date": date, "content_type": "circular", "language": "en", "issuer": "CBDT"})
              for num, date, content in data]
    for num, _, _ in data:
        print(f"  -> {num}")
    save_chunks(chunks, "cbdt")
    return chunks


def extract_gst():
    print("\nExtracting GST Provisions...")
    data = [
        ("Sec 9 CGST",    "Levy and collection",            "GST levied on all intra-state supplies at rates 0, 5, 12, 18, 28 percent. Composition scheme for turnover up to Rs 1.5 crore at 1 percent manufacturer, 5 percent restaurant, 1 percent trader. Reverse charge on specified services."),
        ("Sec 16 CGST",   "Input Tax Credit eligibility",   "ITC available when tax invoice held, goods or services received, tax paid to government, and return filed. ITC not available on motor vehicles for persons up to 13 seats, food and beverages, beauty treatment, health services, club membership."),
        ("Sec 17(5) CGST","Blocked credits",                "ITC blocked on motor vehicles for personal use, food beverages outdoor catering, beauty treatment, health services, club membership, rent-a-cab, life and health insurance, travel benefits to employees, works contract for immovable property."),
        ("Sec 74 CGST",   "Tax determination fraud cases",  "Notice for tax not paid due to fraud or suppression: pay tax plus interest plus penalty equal to 100 percent of tax. Voluntary disclosure before notice reduces penalty to 15 percent. After notice but before order penalty is 25 percent."),
        ("Sec 10 CGST",   "Composition scheme",             "Eligible if turnover up to Rs 1.5 crore. Rates: 1 percent manufacturers and traders, 5 percent restaurant. Cannot make inter-state supply. Cannot supply exempt goods. Cannot collect GST from customers. File quarterly CMP-08."),
        ("Rule 36(4)",    "ITC matching rule",              "ITC cannot exceed 105 percent of eligible ITC appearing in GSTR-2B. Excess ITC must be reversed with interest at 18 percent per annum. Annual reconciliation mandatory in GSTR-9C for taxpayers with turnover above Rs 5 crore."),
        ("Sec 129 CGST",  "Detention and confiscation",     "Goods in transit detained if not accompanied by valid documents. Penalty for detention 200 percent of tax for taxable goods, 2 percent of value for exempt goods. Confiscation if intent to evade tax established by proper officer."),
        ("GSTR Filing",   "Return filing obligations",      "GSTR-1 monthly for turnover above Rs 5 crore or quarterly under QRMP scheme. GSTR-3B monthly self-assessed return with tax payment. GSTR-9 annual return by December 31. GSTR-9C reconciliation statement if turnover above Rs 5 crore."),
    ]
    chunks = [make_chunk(f"{sec} {title} CGST Act 2017: {content}", "gst",
              {"section": sec, "title": title, "content_type": "statute", "language": "en", "act": "CGST Act 2017"})
              for sec, title, content in data]
    for sec, title, _ in data:
        print(f"  -> {sec}: {title}")
    save_chunks(chunks, "gst")
    return chunks


def extract_itat():
    print("\nExtracting Case Law...")
    data = [
        ("CIT v Calcutta Knitwears",          "(2014) 362 ITR 673 SC",        "Supreme Court", "2014", "Sec 10(2A)",         "Share of profit from firm is fully exempt under section 10(2A) even if firm paid tax at lower rate. Exemption is absolute and not conditional on rate of tax paid by firm."),
        ("Pr CIT v Wipro Ltd",                "(2022) 141 taxmann 285 SC",    "Supreme Court", "2022", "Sec 10A",            "Section 10A deduction is deduction from business income at head level, not from total income. Carry forward of business loss after 10A deduction is permissible."),
        ("CIT v Bombay Burmah Trading",       "(1986) 161 ITR 386 SC",        "Supreme Court", "1986", "Sec 37",             "Expenditure incurred wholly and exclusively for business purposes is allowable under section 37 even if it results in some benefit to third party. Test is dominant purpose of expenditure."),
        ("Vodafone International v UOI",      "(2012) 341 ITR 1 SC",          "Supreme Court", "2012", "Sec 9 Transfer",     "Transfer of shares of foreign company holding Indian assets is not taxable in India. No look-through principle in absence of specific legislation. Led to Explanation 5 to Section 9(1)(i)."),
        ("CIT v Kelvinator of India",         "(2010) 320 ITR 561 SC",        "Supreme Court", "2010", "Sec 147",            "Reassessment requires tangible material showing income escaped assessment. Mere change of opinion by AO is not sufficient ground for reopening. AO must have reason to believe, not merely suspect."),
        ("ACIT v Rajesh Jhaveri",             "(2007) 291 ITR 500 SC",        "Supreme Court", "2007", "Sec 147 148",        "At stage of issue of notice under section 148, AO need not establish that income has actually escaped. Reasonable belief sufficient. Objections to be decided before proceeding with reassessment."),
        ("DIT v Infrasoft Ltd",               "(2014) 220 Taxman 273 Del HC", "Delhi HC",      "2014", "Sec 9(1)(vi) Royalty","Payment for use of software is not royalty. Software is a product not a copyright. Not taxable as royalty under section 9(1)(vi) unless copyright in software itself is transferred to user."),
        ("CIT v Shriram Transport Finance",   "(2011) 339 ITR 94 Mad HC",     "Madras HC",     "2011", "Sec 194A",           "TDS under section 194A on interest paid by NBFC to depositors. Threshold limit applies per depositor per branch per year. Branch-wise computation of threshold is permissible."),
    ]
    chunks = [make_chunk(f"Case Law: {case} {citation} Court: {court} Year: {year} Section: {sec} Held: {ratio}",
              "itat", {"case_name": case, "citation": citation, "court": court, "year": year, "section": sec, "content_type": "judgment", "language": "en"})
              for case, citation, court, year, sec, ratio in data]
    for case, _, _, year, _, _ in data:
        print(f"  -> {case} ({year})")
    save_chunks(chunks, "itat")
    return chunks


def extract_budget():
    print("\nExtracting Budget Amendments...")
    data = [
        ("Budget 2024-25", "Sec 115BAC", "New regime slabs revised",       "New regime slabs: Nil up to Rs 3 lakh, 5 percent 3-7 lakh, 10 percent 7-10 lakh, 15 percent 10-12 lakh, 20 percent 12-15 lakh, 30 percent above 15 lakh. Standard deduction increased to Rs 75000. NPS employer deduction raised to 14 percent."),
        ("Budget 2024-25", "Sec 112A",   "LTCG rate increased",             "LTCG on listed equity and equity MF increased from 10 to 12.5 percent. Exemption limit increased from Rs 1 lakh to Rs 1.25 lakh. Indexation benefit removed for all assets including property from July 23 2024."),
        ("Budget 2024-25", "Sec 111A",   "STCG rate increased",             "STCG on listed equity shares and equity-oriented mutual funds increased from 15 to 20 percent. Applicable where STT paid on recognized stock exchange. Effective from July 23 2024."),
        ("Budget 2024-25", "Sec 80CCD2","NPS employer deduction raised",    "Employer contribution to NPS deductible up to 14 percent of salary for all employees including private sector. Earlier limit was 10 percent for private sector. Additional employee contribution under 80CCD(1B) Rs 50000 continues."),
        ("Budget 2024-25", "Sec 43B_h", "MSME payment deduction",           "Payment to micro and small enterprises within 45 days with agreement or 15 days without agreement. Unpaid amount disallowed until actual payment. Effective AY 2024-25. Medium enterprises not covered under this provision."),
        ("Budget 2023-24", "Sec 54 54F","Capital gains exemption cap",      "Cap of Rs 10 crore on exemption under sections 54 and 54F for reinvestment in residential property. Effective AY 2024-25. Amount exceeding Rs 10 crore taxable as long term capital gain."),
        ("Budget 2023-24", "Sec 115BAC","New regime as default",            "New tax regime made default from AY 2024-25. Taxpayers must opt for old regime by filing Form 10-IEA before due date. Salaried employees can switch every year. Business or professional taxpayers can switch only once from old to new."),
        ("Budget 2022-23", "Sec 139_8A","Updated return ITR-U",             "Updated return under section 139(8A) within 2 years of end of assessment year. Additional tax 25 percent if filed within 12 months, 50 percent if filed between 12 to 24 months. Cannot be filed for loss returns or refund claims."),
    ]
    chunks = [make_chunk(f"{budget} {title} ({sec}): {content}", "budget",
              {"budget": budget, "section": sec, "amendment": title, "content_type": "budget_amendment", "language": "en"})
              for budget, sec, title, content in data]
    for _, _, title, _ in data:
        print(f"  -> {title}")
    save_chunks(chunks, "budget")
    return chunks


# ── Review / Approve / Index ──────────────────────────────────────────────────

def show_review_queue():
    files = list(Path("review_queue").glob("*.json"))
    if not files:
        print("Review queue is empty.")
        return
    total = 0
    for f in files:
        chunks = json.loads(f.read_text(encoding="utf-8"))
        pending = [c for c in chunks if c["status"] == "pending"]
        total += len(pending)
        print(f"  {f.name}: {len(pending)} chunks")
        if pending:
            print(f"    Sample: {pending[0]['text'][:120]}...")
    print(f"\n  Total pending: {total} chunks")


def approve_all():
    files = list(Path("review_queue").glob("*.json"))
    if not files:
        print("Nothing to approve.")
        return
    total = 0
    for f in files:
        chunks = json.loads(f.read_text(encoding="utf-8"))
        for c in chunks:
            c["status"] = "approved"
            c["approved_at"] = datetime.now().isoformat()
        dest = Path("approved") / f.name
        dest.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        f.unlink()
        total += len(chunks)
        print(f"  Approved {len(chunks)} chunks from {f.name}")
    print(f"\n  Total approved: {total} chunks")


def index_to_opensearch():
    from opensearchpy import OpenSearch
    files = list(Path("approved").glob("*.json"))
    if not files:
        print("No approved chunks. Run approve first.")
        return
    client = OpenSearch(OPENSEARCH_URL, verify_certs=False, ssl_show_warn=False)
    index  = OPENSEARCH_INDEX
    if not client.indices.exists(index=index):
        client.indices.create(index=index, body={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {"properties": {
                "text":         {"type": "text", "analyzer": "english"},
                "source":       {"type": "keyword"},
                "content_type": {"type": "keyword"},
                "section":      {"type": "keyword"},
                "year":         {"type": "keyword"},
                "language":     {"type": "keyword"},
            }}
        })
        print(f"Created index '{index}'")
    total = 0
    for f in files:
        chunks = json.loads(f.read_text(encoding="utf-8"))
        approved = [c for c in chunks if c["status"] == "approved"]
        for chunk in approved:
            client.index(index=index, body={
                "text":         chunk["text"],
                "source":       chunk["source"],
                "metadata":     chunk["metadata"],
                "content_type": chunk["metadata"].get("content_type", "unknown"),
                "section":      chunk["metadata"].get("section_number") or chunk["metadata"].get("section", ""),
                "year":         chunk["metadata"].get("year", ""),
                "language":     chunk["metadata"].get("language", "en"),
                "indexed_at":   datetime.now().isoformat(),
            }, id=chunk["id"], params={"refresh": "true"})
            total += 1
        f.rename(Path("logs") / f.name)
        print(f"  Indexed {len(approved)} chunks from {f.name}")
    print(f"\n  Total indexed: {total} chunks into '{index}'")


# ── CLI ───────────────────────────────────────────────────────────────────────

EXTRACTORS = {
    "it_act": extract_it_act,
    "cbdt":   extract_cbdt,
    "gst":    extract_gst,
    "itat":   extract_itat,
    "budget": extract_budget,
}

def main():
    parser = argparse.ArgumentParser(description="TaxMind KB Pipeline")
    parser.add_argument("command", choices=["extract", "review", "approve", "index", "run-all"])
    parser.add_argument("--source", choices=list(EXTRACTORS.keys()))
    args = parser.parse_args()

    if args.command == "extract":
        if args.source:
            EXTRACTORS[args.source]()
        else:
            for name, fn in EXTRACTORS.items():
                try:
                    fn()
                except Exception as e:
                    print(f"  {name} failed: {e}")
    elif args.command == "review":
        show_review_queue()
    elif args.command == "approve":
        approve_all()
    elif args.command == "index":
        index_to_opensearch()
    elif args.command == "run-all":
        for name, fn in EXTRACTORS.items():
            try:
                fn()
            except Exception as e:
                print(f"  {name} failed: {e}")
        approve_all()
        index_to_opensearch()
        print("\nPipeline complete!")

if __name__ == "__main__":
    main()