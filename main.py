import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List
import numpy as np
import cv2
import pytesseract
import re
import os
from datetime import datetime

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = FastAPI(title="CompliX Advanced LMPC 2011 Engine")

# Enable Cross-Origin Resource Sharing for your Netlify web domain app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. LOCAL DATA STORAGE REPOSITORY ---
DATABASE_URL = "sqlite:///./audit_ledger.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class InspectionAuditLog(Base):
    __tablename__ = "inspection_audits"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    barcode = Column(String, index=True)
    product_name = Column(String)
    status = Column(String)
    infractions = Column(Text)
    pdf_path = Column(String)

Base.metadata.create_all(bind=engine)

# --- 2. EXPANDED GOVERNMENT BARCODE REGISTRY MATRIX ---
REGISTERED_PRODUCT_REGISTRY = {
    "8901058002316": {
        "product_name": "Parle Glucose Biscuits", 
        "registered_mrp": 10.00, 
        "registered_net_qty": "100g", 
        "manufacturer": "Parle Biscuits Pvt Ltd, Mumbai"
    },
    "8901491101836": {
        "product_name": "Lays Spiced Potato Chips", 
        "registered_mrp": 20.00, 
        "registered_net_qty": "50g", 
        "manufacturer": "SnackFoods India, New Delhi"
    },
    "8901058895628": {
        "product_name": "Maggi 2-Minute Noodles", 
        "registered_mrp": 14.00, 
        "registered_net_qty": "70g", 
        "manufacturer": "Nestlé India Limited, Gurugram"
    },
    "8901262010124": {
        "product_name": "Amul Pasteurized Butter", 
        "registered_mrp": 56.00, 
        "registered_net_qty": "100g", 
        "manufacturer": "GCMMF Ltd, Anand, Gujarat"
    },
    "5449000000996": {
        "product_name": "Coca-Cola Original Taste 250ml", 
        "registered_mrp": 20.00, 
        "registered_net_qty": "250ml", 
        "manufacturer": "Hindustan Coca-Cola Beverages, Bengaluru"
    }
}

# --- 3. DYNAMIC PDF VIOLATION FORMS COMPILER ---
def build_enforcement_notice(barcode: str, product_name: str, manufacturer: str, violations: List[str]) -> str:
    os.makedirs("./notices", exist_ok=True)
    filename = f"./notices/NOTICE_{barcode}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], textColor=colors.HexColor('#990000'), spaceAfter=12)
    normal_style = styles['Normal']
    
    story.append(Paragraph("GOVERNMENT OF INDIA • MINISTRY OF CONSUMER AFFAIRS", normal_style))
    story.append(Paragraph("LEGAL METROLOGY ENFORCEMENT NOTICE (RULE 6 VIOLATION)", title_style))
    story.append(Paragraph(f"<b>Audit Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 15))
    
    body_text = f"""
    This formal non-compliance notice is issued to the registered manufacturer/packer <b>{manufacturer}</b> 
    under the statutory framework of the <b>Legal Metrology (Packaged Commodities) Rules, 2011</b>. 
    Digital scanning validation of product unit <b>{product_name}</b> (EAN Barcode: {barcode}) has flagged 
    clear, actionable visual configuration infractions on the physical container wrap.
    <br/><br/>
    The automated compliance validation logging engine recorded the following discrepancies:
    """
    story.append(Paragraph(body_text, normal_style))
    story.append(Spacer(1, 10))
    
    for violation in violations:
        story.append(Paragraph(f"• <font color='red'><b>CRITICAL INFRACTION:</b></font> {violation}", normal_style))
        story.append(Spacer(1, 6))
        
    closing = """
    <br/><br/>
    <b>Directives:</b> You are required to submit an explanation or pull the non-compliant batches from physical 
    and electronic retail retail spaces within fifteen (15) working days from notice delivery.
    <br/><br/>
    <i>Generated via automated image analysis execution by Project CompliX Core Pipeline.</i>
    """
    story.append(Paragraph(closing, normal_style))
    
    doc.build(story)
    return filename

# --- 4. DATA COMPLIANCE SCHEMAS ---
class RuleCheckStatus(BaseModel):
    check_name: str
    status: str
    details: str

class BarcodeComplianceReport(BaseModel):
    barcode_found: str
    product_identified: str
    overall_compliance: str
    executed_checks: List[RuleCheckStatus]
    pdf_download_url: str

# --- 5. MAIN CORE ALGORITHM ROUTE ---
@app.post("/api/v1/compliance/verify", response_model=BarcodeComplianceReport)
async def verify_packaged_commodity(barcode: str = Query(...), file: UploadFile = File(...)):
    db = SessionLocal()
    try:
        # Step A: Validate against the database registry mapping dictionary
        if barcode not in REGISTERED_PRODUCT_REGISTRY:
            raise HTTPException(status_code=404, detail="Barcode string matches no registered legal filings.")
            
        product_metadata = REGISTERED_PRODUCT_REGISTRY[barcode]
        
        # Step B: Read camera data payload stream directly into OpenCV matrix arrays
        file_bytes = await file.read()
        np_array = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        # Optimize rendering framework matrix by shifting canvas vectors to grayscales
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        combined_text = pytesseract.image_to_string(gray).lower()

        checks_log = []
        violation_strings = []
        is_fully_compliant = True

        # Rule Check 1: Retail Price Verification [Rule 6(1)(da)]
        found_prices = [float(x) for x in re.findall(r'(?:rs\.?|mrp)\s*(\d+(?:\.\d+)?)', combined_text)]
        if found_prices:
            actual_printed_price = found_prices[0]
            if actual_printed_price > product_metadata["registered_mrp"]:
                is_fully_compliant = False
                msg = f"Overcharging infraction! Printed Price Rs.{actual_printed_price} exceeds legal filing (Rs.{product_metadata['registered_mrp']})."
                violation_strings.append(msg)
                checks_log.append(RuleCheckStatus(check_name="Rule 6(1)(da) - MRP Integrity", status="VIOLATION", details=msg))
            else:
                checks_log.append(RuleCheckStatus(check_name="Rule 6(1)(da) - MRP Integrity", status="PASSED", details=f"Printed price Rs.{actual_printed_price} is within bounds."))
        else:
            is_fully_compliant = False
            msg = "Maximum Retail Price (MRP) text statement was missing or unreadable on package labels."
            violation_strings.append(msg)
            checks_log.append(RuleCheckStatus(check_name="Rule 6(1)(da) - MRP Presence", status="VIOLATION", details=msg))

        # Rule Check 2: Tax Inclusion Phrasing [Rule 6(1)(da)]
        if "incl" not in combined_text and "tax" not in combined_text:
            is_fully_compliant = False
            msg = "Mandatory suffix 'Incl. of all taxes' description string was absent from packaging price print."
            violation_strings.append(msg)
            checks_log.append(RuleCheckStatus(check_name="Rule 6(1)(da) - Suffix Framework", status="VIOLATION", details=msg))
        else:
            checks_log.append(RuleCheckStatus(check_name="Rule 6(1)(da) - Suffix Framework", status="PASSED", details="Statutory tax statements verified."))

        # Rule Check 3: Manufacturing Date Presence & Validation [Rule 6(1)(d)]
        # This scans for standard date patterns like MM/YYYY, MM-YYYY, or string names like "Mfg Date" [03/2026]
        date_pattern_match = re.search(r'(?:mfg|pkd|packed|date|mfd)\b.*?(\d{2}[/\-]\d{4}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[/\-]\d{2,4})', combined_text)
        if date_pattern_match:
            extracted_date_string = date_pattern_match.group(1)
            checks_log.append(RuleCheckStatus(
                check_name="Rule 6(1)(d) - Packing Date Declaration", 
                status="PASSED", 
                details=f"Valid chronological manufacture timeline mark detected: [{extracted_date_string}]."
            ))
        else:
            is_fully_compliant = False
            msg = "Mandatory Month & Year of packing/manufacture declaration [Rule 6(1)(d)] was not detected on visual surface layout."
            violation_strings.append(msg)
            checks_log.append(RuleCheckStatus(check_name="Rule 6(1)(d) - Packing Date Declaration", status="VIOLATION", details=msg))

        # Rule Check 4: Legal Quantity Matching [Rule 6(1)(c)]
        expected_qty = product_metadata["registered_net_qty"].lower()
        if expected_qty in combined_text:
            checks_log.append(RuleCheckStatus(check_name="Rule 6(1)(c) - Net Quantity Consistency", status="PASSED", details=f"Net content weight declaration ({product_metadata['registered_net_qty']}) matches database values."))
        else:
            is_fully_compliant = False
