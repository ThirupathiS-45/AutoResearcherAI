import sys
import os
import requests
import streamlit as st

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing project modules
from agents.search_agent import search_arxiv
from agents.codegen_agent import generate_code_from_paper
from agents.pdf_qa_agent import load_pdf_and_build_qa_chain
from utils.pdf_extractor import extract_text_from_pdf

# ---------------------- Page Config ----------------------
st.set_page_config(page_title="AutoResearcherAI", layout="wide")

# ---------------------- Centered Title ----------------------
st.markdown(
    "<h1 style='text-align: center;'>🤖 AutoResearcherAI — Research to Code Generator</h1>",
    unsafe_allow_html=True
)
st.markdown("🔍 Enter a research topic, browse papers, and generate working code.")

# ---------------------- Search Input ----------------------
query = st.text_input("🧠 Research Topic", value="vision transformers")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🔍 Search Papers"):
        with st.spinner("Searching arXiv..."):
            papers = search_arxiv(query, max_results=5)

        if not papers:
            st.error("❌ No papers found.")
        else:
            st.session_state["papers"] = papers
            st.session_state["selected_index"] = 0

# ---------------------- Paper Selection ----------------------
if "papers" in st.session_state:
    st.markdown("---")
    papers = st.session_state["papers"]
    paper_titles = [f"{i+1}. {p['title'][:100]}" for i, p in enumerate(papers)]

    selected_index = st.selectbox("📄 Select a paper", range(len(papers)), format_func=lambda i: paper_titles[i])
    paper = papers[selected_index]

    st.subheader("📌 Title")
    st.write(paper["title"])

    with st.expander("📚 Abstract", expanded=True):
        st.write(paper["summary"])

    st.subheader("🔗 PDF Link")
    st.markdown(f"[📥 Download PDF]({paper['pdf_url']})")

    # ---------------------- Code Generation ----------------------
    st.markdown("---")
    st.subheader("🧠 AI Code Generation")

    if st.button("🚀 Generate Code from This Paper"):
        with st.spinner("📥 Downloading and analyzing paper..."):
            pdf_path = "temp.pdf"
            response = requests.get(paper["pdf_url"])
            with open(pdf_path, "wb") as f:
                f.write(response.content)

            paper_text = extract_text_from_pdf(pdf_path)

        if paper_text:
            with st.spinner("🤖 Generating code using Gemini..."):
                code = generate_code_from_paper(paper_text[:3000])

            if code:
                with st.expander("✅ View AI-Generated Code", expanded=True):
                    st.code(code, language="python")

                os.makedirs("generated", exist_ok=True)
                code_path = os.path.join("generated", "generated_code.py")
                with open(code_path, "w", encoding="utf-8") as f:
                    f.write(code)

                st.success("💾 Code saved to `generated/generated_code.py`")
                st.download_button("⬇️ Download Code", code, file_name="generated_code.py", mime="text/x-python")
            else:
                st.error("❌ Failed to generate code.")
        else:
            st.error("❌ Failed to extract PDF content.")

    # ---------------------- Paper Q&A with LangChain + Gemini ----------------------
    st.markdown("---")
    st.subheader("🤖 Ask Questions from This Paper")

    if st.button("📄 Enable Paper Q&A"):
        with st.spinner("🔍 Loading and embedding paper..."):
            pdf_path = "temp.pdf"
            response = requests.get(paper["pdf_url"])
            with open(pdf_path, "wb") as f:
                f.write(response.content)

            qa_chain = load_pdf_and_build_qa_chain(pdf_path, os.getenv("GOOGLE_API_KEY"))
            st.session_state["qa_chain"] = qa_chain
            st.success("✅ Ready to answer questions!")

    if "qa_chain" in st.session_state:
        user_question = st.text_input("❓ Ask a question about this paper")
        if user_question:
            with st.spinner("💬 Thinking..."):
                response = st.session_state["qa_chain"]({"query": user_question})
                answer = response["result"]
                st.markdown(f"**💡 Answer:** {answer}")

                with st.expander("📄 Source Snippets"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**📄 Source {i+1}:** {doc.page_content[:300]}...")
