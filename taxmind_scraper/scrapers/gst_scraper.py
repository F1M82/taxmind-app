import hashlib
import json
from pathlib import Path
from typing import Iterator
from loguru import logger
from bs4 import BeautifulSoup
from utils.http_client import EthicalHttpClient
from config.settings import settings

CBIC_CIRCULARS_URL = "https://www.cbic.gov.in/resources//htdocs-cbec/gst/notfctn-cgst-eng.pdf"
CBIC_BASE = "https://cbic-gst.gov.in"

GST_SECTIONS = [
    {"section": "2", "title": "Definitions", "act": "CGST", "content": "Section 2 CGST Act 2017: Defines key terms including aggregate turnover, business, capital goods, casual taxable person, composite supply, continuous supply, exempt supply, fixed establishment, goods, input, input service, input tax, input tax credit, mixed supply, non-resident taxable person, outward supply, place of business, principal supply, recipient, reverse charge, supplier, supply, tax invoice, taxable person, taxable supply, taxable territory, zero rated supply."},
    {"section": "7", "title": "Scope of Supply", "act": "CGST", "content": "Section 7 CGST Act 2017: Supply includes all forms of supply of goods or services for a consideration in the course of furtherance of business. Schedule I activities treated as supply even without consideration. Schedule II activities classified as supply of goods or services. Schedule III activities neither supply of goods nor services."},
    {"section": "9", "title": "Levy and Collection", "act": "CGST", "content": "Section 9 CGST Act 2017: CGST levied on all intra-state supplies of goods or services at rates not exceeding 20 percent. Section 9(3) reverse charge on notified goods and services. Section 9(4) reverse charge on supplies from unregistered persons to registered persons for notified categories."},
    {"section": "10", "title": "Composition Levy", "act": "CGST", "content": "Section 10 CGST Act 2017: Registered persons with aggregate turnover up to 1.5 crore may opt for composition scheme. Rate 1 percent for manufacturers, 2.5 percent for restaurants, 0.5 percent for traders. Not applicable for inter-state supplies, e-commerce operators, non-resident taxable persons, casual taxable persons."},
    {"section": "16", "title": "Input Tax Credit Eligibility", "act": "CGST", "content": "Section 16 CGST Act 2017: Registered person entitled to ITC on goods or services used in course of business. Four conditions: possession of tax invoice, receipt of goods or services, tax actually paid by supplier, return filed. ITC blocked if depreciation claimed on tax component. Time limit to claim ITC is earlier of due date of September return of next FY or date of annual return."},
    {"section": "17", "title": "Apportionment of ITC", "act": "CGST", "content": "Section 17 CGST Act 2017: ITC not available for goods or services used for exempt supplies. Proportionate ITC for mixed use. Section 17(5) blocked credits: motor vehicles for personal use, food and beverages, outdoor catering, beauty treatment, health services, membership of club, travel benefits to employees, works contract for immovable property, goods or services for personal consumption, goods lost or stolen or destroyed."},
    {"section": "22", "title": "Registration Threshold", "act": "CGST", "content": "Section 22 CGST Act 2017: Every supplier whose aggregate turnover exceeds 40 lakhs for goods or 20 lakhs for services in a financial year shall be liable to register. Threshold 10 lakhs for special category states. Aggregate turnover includes taxable supplies, exempt supplies, inter-state supplies and exports but excludes inward supplies under reverse charge."},
    {"section": "24", "title": "Compulsory Registration", "act": "CGST", "content": "Section 24 CGST Act 2017: Compulsory registration regardless of turnover threshold for: inter-state taxable suppliers, casual taxable persons, persons liable to pay tax under reverse charge, e-commerce operators, input service distributors, persons supplying through e-commerce operators, non-resident taxable persons, persons required to deduct TDS."},
    {"section": "31", "title": "Tax Invoice", "act": "CGST", "content": "Section 31 CGST Act 2017: Registered person supplying taxable goods shall issue tax invoice before or at the time of removal or delivery. For services invoice to be issued within 30 days of supply. Invoice must contain: supplier GSTIN, consecutive serial number, date, recipient details, HSN code, description, quantity, value, taxable value, rate and amount of tax."},
    {"section": "37", "title": "GSTR-1 Outward Supplies", "act": "CGST", "content": "Section 37 CGST Act 2017: Every registered person shall furnish details of outward supplies in GSTR-1. Monthly filers with turnover above 5 crore file by 11th of following month. Quarterly filers with turnover up to 5 crore file by 13th of month following quarter. Details include B2B invoices, B2C large invoices, credit notes, debit notes, exports."},
    {"section": "38", "title": "GSTR-2B Auto-drafted ITC", "act": "CGST", "content": "Section 38 CGST Act 2017: Auto-drafted statement of ITC available to recipient based on suppliers GSTR-1 and GSTR-5. Generated on 14th of each month. Recipient can accept, reject or keep pending. ITC available only on accepted invoices. Mismatch between GSTR-2B and purchase register to be reconciled before claiming ITC."},
    {"section": "39", "title": "GSTR-3B Returns", "act": "CGST", "content": "Section 39 CGST Act 2017: Every registered person shall file monthly return GSTR-3B with summary of outward supplies, ITC claimed, tax payable and tax paid. Monthly filers with turnover above 5 crore file by 20th of following month. Quarterly filers under QRMP scheme file by 22nd or 24th depending on state. Late fee 50 rupees per day, 20 rupees for nil return."},
    {"section": "44", "title": "Annual Return GSTR-9", "act": "CGST", "content": "Section 44 CGST Act 2017: Every registered person except casual taxable person and non-resident taxable person shall file annual return GSTR-9 by 31st December of following financial year. Reconciliation statement GSTR-9C required if turnover exceeds 5 crore. GSTR-9C to be certified by CA or CMA."},
    {"section": "50", "title": "Interest on Late Payment", "act": "CGST", "content": "Section 50 CGST Act 2017: Interest at 18 percent per annum on delayed payment of tax. Interest at 24 percent for wrongful availment and utilization of ITC. Interest calculated on net tax liability after adjusting ITC. Interest to be paid from next day after due date to actual date of payment."},
    {"section": "54", "title": "Refund of Tax", "act": "CGST", "content": "Section 54 CGST Act 2017: Refund application to be filed within 2 years from relevant date. Relevant date for export of goods is date of departure of ship or aircraft. For services export relevant date is date of receipt of payment in convertible foreign exchange. Refund of ITC accumulated due to inverted duty structure or zero rated supplies without payment of tax."},
    {"section": "61", "title": "Scrutiny of Returns", "act": "CGST", "content": "Section 61 CGST Act 2017: Proper officer may scrutinize return to verify correctness. Notice issued to registered person to explain discrepancies. Person to provide explanation within 30 days or extended time. If explanation found acceptable no further action. If discrepancy confirmed demand may be raised under section 73 or 74."},
    {"section": "73", "title": "Demand Non-Fraud Cases", "act": "CGST", "content": "Section 73 CGST Act 2017: Show cause notice for recovery of tax not paid or short paid or erroneously refunded or ITC wrongly availed without fraud. Time limit 3 years from due date of annual return. Penalty 10 percent of tax or 10000 rupees whichever is higher. No penalty if tax, interest and penalty paid within 30 days of show cause notice."},
    {"section": "74", "title": "Demand Fraud Cases", "act": "CGST", "content": "Section 74 CGST Act 2017: Show cause notice where tax not paid due to fraud, wilful misstatement or suppression of facts. Time limit 5 years from due date of annual return. Penalty equal to tax amount. Penalty reduced to 15 percent if paid within 30 days of show cause notice, 25 percent if paid within 30 days of order."},
    {"section": "129", "title": "Detention and Seizure", "act": "CGST", "content": "Section 129 CGST Act 2017: Goods in transit may be detained if not accompanied by valid documents. Penalty 200 percent of tax for taxable goods, 2 percent of value or 25000 rupees for exempt goods. Owner or transporter must pay tax and penalty for release. If not paid within 14 days goods may be confiscated under section 130."},
    {"section": "130", "title": "Confiscation", "act": "CGST", "content": "Section 130 CGST Act 2017: Goods liable to confiscation if supplied without invoice, invoice not matching goods, HSN code mismatch, goods transported without e-way bill, false declaration made. Owner given option to pay fine in lieu of confiscation. Fine cannot exceed market value of goods minus tax chargeable."},
]

GST_CIRCULARS = [
    {"number": "183/15/2022", "date": "2022-12-27", "subject": "ITC on CSR expenditure", "content": "CBDT Circular 183/15/2022-GST: ITC not available on goods or services procured for CSR activities as CSR expenditure is not in the course or furtherance of business. Section 17(5)(h) blocks ITC on goods disposed of as gift or free samples. CSR activities are statutory obligation and not business activity."},
    {"number": "172/04/2022", "date": "2022-03-06", "subject": "E-way bill for intra-state movement", "content": "CBDT Circular 172/04/2022-GST: E-way bill required for intra-state movement of goods where value exceeds threshold notified by respective state government. E-way bill valid for one day for every 200 km. Validity extendable by transporter within 8 hours before or after expiry."},
    {"number": "168/00/2022", "date": "2022-12-17", "subject": "Clarification on various GST issues", "content": "CBDT Circular 168/00/2022-GST: Clarifications on applicability of GST on various services including liquidated damages, notices in lieu of notice period, forfeiture of security deposit, cheque dishonor charges, penalty charges by electricity distribution companies treated as supply of service and liable to GST."},
    {"number": "178/10/2022", "date": "2022-08-03", "subject": "ITC reversal on credit notes", "content": "CBDT Circular 178/10/2022-GST: Supplier issuing credit note must reverse ITC to the extent of credit note. Recipient must reverse ITC if payment not made within 180 days of invoice date. Rule 37 amended to require reversal of ITC proportionate to unpaid consideration."},
    {"number": "196/08/2023", "date": "2023-07-17", "subject": "ITC on warranty replacement", "content": "CBDT Circular 196/08/2023-GST: No ITC reversal required by manufacturer on goods sent to dealer for warranty replacement. Dealer not required to reverse ITC on goods returned under warranty. No GST on warranty replacement within warranty period where no separate consideration charged."},
]

GST_RATES = [
    {"hsn": "0101-0106", "description": "Live animals", "rate": "0", "notes": "Exempt from GST"},
    {"hsn": "1001-1008", "description": "Cereals including rice wheat maize", "rate": "0-5", "notes": "Unbranded cereals exempt, branded attract 5 percent"},
    {"hsn": "2709-2710", "description": "Petroleum products", "rate": "0", "notes": "Outside GST, subject to central excise and state VAT"},
    {"hsn": "3004", "description": "Medicines and pharmaceuticals", "rate": "0-12", "notes": "Life saving drugs nil, others 5-12 percent"},
    {"hsn": "4901-4911", "description": "Books newspapers periodicals", "rate": "0-12", "notes": "Books exempt, printed materials 5-12 percent"},
    {"hsn": "6101-6217", "description": "Apparel and clothing", "rate": "5-12", "notes": "Value up to 1000 rupees 5 percent, above 12 percent"},
    {"hsn": "7201-7229", "description": "Iron and steel products", "rate": "18", "notes": "Most iron and steel products attract 18 percent GST"},
    {"hsn": "8701-8716", "description": "Vehicles and automobiles", "rate": "28", "notes": "Motor vehicles attract 28 percent plus cess"},
    {"hsn": "9954", "description": "Construction services", "rate": "5-18", "notes": "Affordable housing 1 percent, other residential 5 percent, commercial 18 percent"},
    {"hsn": "9963", "description": "Accommodation services", "rate": "0-18", "notes": "Below 1000 rupees exempt, 1000-7500 12 percent, above 7500 18 percent"},
    {"hsn": "9964", "description": "Passenger transport services", "rate": "0-5", "notes": "Railways economy exempt, AC 5 percent, air economy 5 percent, business 12 percent"},
    {"hsn": "9983", "description": "Professional and consulting services", "rate": "18", "notes": "Legal, accounting, management consulting services 18 percent"},
    {"hsn": "9984", "description": "Telecom and internet services", "rate": "18", "notes": "All telecom and internet services 18 percent"},
    {"hsn": "9985", "description": "Support services", "rate": "18", "notes": "Security, cleaning, packing and other support services 18 percent"},
    {"hsn": "9992", "description": "Education services", "rate": "0-18", "notes": "Schools and universities exempt, coaching and vocational 18 percent"},
    {"hsn": "9993", "description": "Health care services", "rate": "0-5", "notes": "Hospital services exempt, health check packages 5 percent"},
]


class GSTScraper:
    def __init__(self):
        self.raw_dir = Path(settings.raw_data_dir) / "gst"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def generate_id(self, content_type: str, identifier: str) -> str:
        key = f"GST-{content_type}-{identifier}"
        hash_val = hashlib.md5(key.encode()).hexdigest()[:10].upper()
        return f"GST-{content_type[:3].upper()}-{hash_val}"

    def get_sections(self) -> Iterator[dict]:
        for section in GST_SECTIONS:
            doc_id = self.generate_id("SECTION", section["section"])
            yield {
                "judgment_id": doc_id,
                "court": "GST",
                "bench": section["act"],
                "sections": [f"section_{section['section']}"],
                "source_url": f"https://cbic-gst.gov.in/gst-act-rules.html",
                "source_site": "cbic_gst",
                "content_type": "statute",
                "title": section["title"],
                "content": section["content"],
            }

    def get_circulars(self) -> Iterator[dict]:
        for circular in GST_CIRCULARS:
            doc_id = self.generate_id("CIRCULAR", circular["number"])
            yield {
                "judgment_id": doc_id,
                "court": "GST",
                "bench": "CBIC",
                "sections": [],
                "source_url": f"https://cbic-gst.gov.in/circulars.html",
                "source_site": "cbic_gst",
                "content_type": "circular",
                "title": circular["subject"],
                "content": circular["content"],
                "circular_number": circular["number"],
                "date": circular["date"],
            }

    def get_rate_schedules(self) -> Iterator[dict]:
        for rate in GST_RATES:
            doc_id = self.generate_id("RATE", rate["hsn"])
            yield {
                "judgment_id": doc_id,
                "court": "GST",
                "bench": "RATE_SCHEDULE",
                "sections": [],
                "source_url": "https://cbic-gst.gov.in/gst-goods-services-rates.html",
                "source_site": "cbic_gst",
                "content_type": "rate_schedule",
                "title": rate["description"],
                "content": f"HSN {rate['hsn']}: {rate['description']}. GST Rate: {rate['rate']} percent. {rate['notes']}",
                "hsn": rate["hsn"],
                "rate": rate["rate"],
            }

    def get_all(self) -> Iterator[dict]:
        logger.info("Loading GST sections...")
        for doc in self.get_sections():
            yield doc
        logger.info("Loading GST circulars...")
        for doc in self.get_circulars():
            yield doc
        logger.info("Loading GST rate schedules...")
        for doc in self.get_rate_schedules():
            yield doc