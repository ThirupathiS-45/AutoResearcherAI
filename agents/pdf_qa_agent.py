import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate


def load_pdf_and_build_qa_chain(pdf_path: str, api_key: str = None):
    """
    Loads a PDF file, builds embeddings using Gemini, and returns a QA chain
    that can answer questions from the paper.

    Parameters:
        pdf_path (str): Path to the local PDF file.
        api_key (str, optional): Google API key for Gemini. Uses env if not provided.

    Returns:
        RetrievalQA: A LangChain Retrieval QA pipeline.
    """

    # ✅ Set the API key if provided
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    # 📄 Step 1: Load and read the PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # ✂️ Step 2: Chunk text for vectorization
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # 🔗 Step 3: Create vector store with Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key  # ✅ correct key name (all lowercase)
    )
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    # 🤖 Step 4: Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
    )

    # 🧠 Step 5: Optional custom prompt
    prompt_template = """
    You are an AI assistant helping summarize and explain academic research papers.
    Based on the following context, answer the question clearly and concisely.

    Context:
    {context}

    Question:
    {question}
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    # 🔁 Step 6: Build the Retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain
