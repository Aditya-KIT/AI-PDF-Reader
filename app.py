import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -------------------- ENV SETUP --------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in environment variables")
    st.stop()


# -------------------- PDF READING --------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# -------------------- TEXT SPLITTING --------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return text_splitter.split_text(text)


# -------------------- CREATE VECTOR STORE --------------------
def create_vector_store(text_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    return vector_store


# -------------------- LOAD VECTOR STORE --------------------
def load_vector_store():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local("faiss_index", embeddings)


# -------------------- FORMAT RETRIEVED DOCS --------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -------------------- BUILD QA CHAIN (LCEL) --------------------
def build_chain(vector_store):

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # ✅ Correct modern model name
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the question as detailed as possible using the provided context.
    If the answer is not available in the context, say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    retriever = vector_store.as_retriever()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain


# -------------------- STREAMLIT APP --------------------
def main():
    st.set_page_config(page_title="Chat PDF (LangChain 2026)", layout="wide")
    st.header("📄 Chat with PDF using Gemini + LangChain")

    # Store vector store in session
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    user_question = st.text_input("Ask a question from the PDF")

    if user_question:
        if st.session_state.vector_store is None:
            st.warning("⚠ Please upload and process a PDF first.")
        else:
            chain = build_chain(st.session_state.vector_store)
            response = chain.invoke(user_question)
            st.write("### 🤖 Reply:")
            st.write(response)

    with st.sidebar:
        st.title("📂 Upload PDF")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF")
                return

            with st.spinner("Processing PDFs..."):

                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No readable text found in PDF.")
                    return

                text_chunks = get_text_chunks(raw_text)

                vector_store = create_vector_store(text_chunks)

                st.session_state.vector_store = vector_store

                st.success("✅ PDF processed successfully!")


if __name__ == "__main__":
    main()