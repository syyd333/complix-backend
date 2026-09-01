import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List
import numpy as np
import cv2
import easyocr
import re
import os
from datetime import datetime

# Database imports
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# PDF Generation imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Initialize FastAPI app
app = FastAPI(title="CompliX Ultimate LMPC 2011 Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. DATABASE CONFIGURATION ---
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
reader = easyocr.Reader(['en'])

REGISTERED_PRODUCT_REGISTRY = {
    "8901058002316": {"product_name": "Premium Glucose Biscuits", "registered_mrp": 10.00, "registered_net_qty": "100g", "manufacturer": "Parle Biscuits Pvt Ltd"},
    "8901491101836": {"product_name": "Spiced Potato Chips", "registered_mrp": 20.00, "registered_net_qty": "50g", "manufacturer": "SnackFoods India"}
}

# --- 2. AUTOMATED PDF LEGAL NOTICE GENERATOR ---
def build_enforcement_notice(barcode: str, product_name: str, manufacturer: str, violations: List[str]) -> str:
    os.makedirs("./notices", exist_ok=True)
    filename = f"./notices/NOTICE_{barcode}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom Styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], textColor=colors.HexColor('#990000'), spaceAfter=12)
    normal_style = styles['Normal']
    
    story.append(Paragraph("OFFICIAL LEGAL METROLOGY ENFORCEMENT NOTICE", title_style))
    story.append(Paragraph(f"<b>Issued Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 12))
    
    body_text = f"""
    This statutory warning notice is issued to <b>{manufacturer}</b> regarding verified packaging compliance violations detected 
    under the <b>Legal Metrology (Packaged Commodities) Rules, 2011</b> for product item <b>{product_name}</b> (Barcode: {barcode}).
    <br/><br/>
    During automated digital scan verification, the following systematic configuration non-compliances were identified:
    """
    story.append(Paragraph(body_text, normal_style))
    story.append(Spacer(1, 10))
    
    for violation in violations:
        story.append(Paragraph(f"• <font color='red'><b>VIOLATION:</b></font> {violation}", normal_style))
        story.append(Spacer(1, 6))
        
    closing = """
    <br/><br/>
    You are hereby directed to rectify these labeling errors or submit a clarifying response to the undersigned controller 
    within fifteen (15) business working days of notice delivery, failing which standard penalty prosecution files will be initiated.
    <br/><br/>
    <b>Authorized By:</b> Legal Metrology Enforcement Desk (Team CompliX System)
    """
    story.append(Paragraph(closing, normal_style))
    
    doc.build(story)
    return filename

# --- 3. API ROUTER INFRASTRUCTURE ---
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

@app.post("/api/v1/compliance/verify", response_model=BarcodeComplianceReport)
async def verify_packaged_commodity(barcode: str = Query(...), file: UploadFile = File(...)):
    db = SessionLocal()
    try:
        if barcode not in REGISTERED_PRODUCT_REGISTRY:
            raise HTTPException(status_code=404, detail="Barcode registry parameters missing.")
            
        product_metadata = REGISTERED_PRODUCT_REGISTRY[barcode]
        file_bytes = await file.read()
        np_array = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        ocr_results = reader.readtext(img)
        detected_text_list = [res[1] for res in ocr_results]
        combined_text = " ".join(detected_text_list).lower()

        checks_log = []
        violation_strings = []
        is_fully_compliant = True

        # Rule Check 1: MRP Verification
        found_prices = [float(x) for x in re.findall(r'(?:rs\.?|mrp)\s*(\d+(?:\.\d+)?)', combined_text)]
        if found_prices and found_prices[0] > product_metadata["registered_mrp"]:
            is_fully_compliant = False
            msg = f"Printed Price Rs.{found_prices[0]} exceeds registered value of Rs.{product_metadata['registered_mrp']}."
            violation_strings.append(msg)
            checks_log.append(RuleCheckStatus(check_name="Rule 6(1)(da) - MRP Pricing", status="VIOLATION", details=msg))
        else:
            checks_log.append(RuleCheckStatus(check_name="Rule 6(1)(da) - MRP Pricing", status="PASSED", details="Pricing variables within accepted bounds."))

        # Rule Check 2: Tax Inclusion Phrasing
        if "incl" not in combined_text and "tax" not in combined_text:
            is_fully_compliant = False
            msg = "Mandatory phrase suffix 'Incl. of all taxes' declaration was missing from packaging text layout."
            violation_strings.append(msg)
            checks_log.append(RuleCheckStatus(check_name="Rule 6(1)(da) - Tax Statements", status="VIOLATION", details=msg))
        else:
            checks_log.append(RuleCheckStatus(check_name="Rule 6(1)(da) - Tax Statements", status="PASSED", details="Statutory tax statements verified."))

        final_verdict = "PASSED" if is_fully_compliant else "FAILED"
        
        # 4. Generate the PDF notice if any violations were found
        generated_pdf = ""
        pdf_url = ""
        if not is_fully_compliant:
            generated_pdf = build_enforcement_notice(
                barcode, product_metadata["product_name"], product_metadata["manufacturer"], violation_strings
            )
            pdf_url = f"http://127.0.0{barcode}"

        # 5. Commit record permanently to the SQLite database
        audit_record = InspectionAuditLog(
            barcode=barcode,
            product_name=product_metadata["product_name"],
            status=final_verdict,
            infractions="; ".join(violation_strings),
            pdf_path=generated_pdf
        )
        db.add(audit_record)
        db.commit()

        return BarcodeComplianceReport(
            barcode_found=barcode,
            product_identified=product_metadata["product_name"],
            overall_compliance=final_verdict,
            executed_checks=checks_log,
            pdf_download_url=pdf_url
        )
    finally:
        db.close()

@app.get("/api/v1/compliance/download-notice")
def download_notice(barcode: str):
    db = SessionLocal()
    record = db.query(InspectionAuditLog).filter(InspectionAuditLog.barcode == barcode).order_by(InspectionAuditLog.id.desc()).first()
    db.close()
    if record and record.pdf_path and os.path.exists(record.pdf_path):
        return FileResponse(record.pdf_path, media_type='application/pdf', filename=os.path.basename(record.pdf_path))
    raise HTTPException(status_code=404, detail="No violation notice report found for this product.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
