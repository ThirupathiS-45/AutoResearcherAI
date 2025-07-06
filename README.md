# 🤖 AutoResearcherAI — Research to Code Generator

AutoResearcherAI is a powerful web application that bridges the gap between **AI research and implementation**. It enables users to:

- 🔍 Search **arXiv** papers based on a research topic  
- 📥 Download and analyze research paper PDFs  
- 🧠 Generate working **Python code** using **Google Gemini**  
- ❓ Ask intelligent questions from the paper using **LangChain + Gemini Q&A**  

This tool is especially useful for **researchers**, **data scientists**, and **AI enthusiasts** who want to accelerate their understanding of papers and build prototypes quickly.

---

## 🛠 Features

- 🔎 **arXiv Search**: Retrieve top papers by topic  
- 📄 **PDF Extraction**: Parse and analyze the contents  
- 🤖 **Code Generation**: Use Gemini LLM to generate runnable code  
- 💬 **Paper Q&A**: Ask contextual questions with LangChain’s retrieval-based QA  
- 📦 **Download AI-Generated Code** in one click  

---

## 🧩 Tech Stack

- [Streamlit](https://streamlit.io/) – UI framework  
- [LangChain](https://www.langchain.com/) – LLM orchestration  
- [Google Gemini](https://ai.google.dev/) – LLM & Embedding provider  
- [FAISS](https://github.com/facebookresearch/faiss) – Vector similarity search  
- [arXiv API](https://arxiv.org/help/api/) – Paper metadata  
- Python 3.9+  

---

## 📂 Folder Structure

```
AutoResearcherAI/
│
├── app/
│   └── streamlit_app.py             # Main Streamlit app
│
├── agents/
│   ├── search_agent.py              # arXiv search
│   ├── codegen_agent.py             # Gemini code generation
│   └── pdf_qa_agent.py              # PDF QA chain using LangChain
│
├── utils/
│   └── pdf_extractor.py             # PDF text extraction
│
├── generated/
│   └── generated_code.py            # Auto-saved generated code
│
├── requirements.txt
└── README.md
```
---

## ✅ Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/AutoResearcherAI.git
cd AutoResearcherAI
```

2. **Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Set Your Environment Variable**

```bash
# .env or system-level
export GOOGLE_API_KEY=your_gemini_api_key
```

5. **Run the App**

```bash
streamlit run app/streamlit_app.py
```

---

## 🔐 Environment Variable

| Variable         | Description                        |
|------------------|------------------------------------|
| `GOOGLE_API_KEY` | Your Gemini API key from Google AI |

---

## 📦 Dependencies

Sample `requirements.txt`:

```
streamlit
langchain
langchain-google-genai
faiss-cpu
PyMuPDF
requests
python-dotenv
```

> If you encounter issues with `faiss`, ensure you're using `faiss-cpu` and not `faiss` on incompatible numpy versions.

---

## 🙌 Acknowledgements

- Google Gemini for API access  
- LangChain for building LLM pipelines  
- Streamlit for quick app prototyping  
- arXiv API for research paper data  
---

## ⭐ Show Your Support

If you find this project useful, please ⭐ star the repo!
