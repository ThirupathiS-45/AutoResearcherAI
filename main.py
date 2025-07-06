# main.py

from agents.search_agent import search_arxiv
from agents.codegen_agent import generate_code_from_paper
from utils.pdf_extractor import extract_text_from_pdf

import requests

def download_pdf(pdf_url, filename="paper.pdf"):
    try:
        response = requests.get(pdf_url)
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ PDF downloaded: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Failed to download PDF: {e}")
        return None

# 🔍 Search papers
query = "vision transformers"
papers = search_arxiv(query)

if not papers:
    print("❌ No papers found.")
else:
    print(f"\n📄 Top result: {papers[0]['title']}")
    print("🔗 PDF:", papers[0]['pdf_url'])

    pdf_file = download_pdf(papers[0]["pdf_url"])

    if pdf_file:
        full_text = extract_text_from_pdf(pdf_file)

        if full_text:
            print("📚 Extracted text... generating code...\n")
            code = generate_code_from_paper(full_text[:3000])  # Truncate for prompt safety

            if code:
                print("✅ Code Generated:\n")
                print(code)
            else:
                print("❌ Code generation failed.")
        else:
            print("❌ PDF text extraction failed.")
