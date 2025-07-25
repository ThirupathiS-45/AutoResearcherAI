# utils/pdf_extractor.py

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return None
