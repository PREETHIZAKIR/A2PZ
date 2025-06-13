import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# === CONFIGURE YOUR GOOGLE API KEY ===
GOOGLE_API_KEY = "AIzaSyDK5hw1kVgVBgJJc64SHH7T9pJOWM2U2lk"  # 🔐 Replace with actual key

# === SETUP Gemini LLM and Embeddings ===
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # required model for embeddings
    google_api_key=GOOGLE_API_KEY
)

# === STREAMLIT UI ===
st.set_page_config(page_title="📄 PDF Q&A with RAG", layout="centered")
st.title("🔍 Ask Questions from Your PDF using RAG")

uploaded_pdf = st.file_uploader("📤 Upload a PDF file", type=["pdf"])
query = st.text_input("💬 Ask a question about the PDF:")

if uploaded_pdf:
    # Save uploaded file to a temporary location
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    # Load and split PDF
    loader = PyPDFLoader("temp_uploaded.pdf")
    pages = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    # Create QA Chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # If user submitted a query, run QA
    if query:
        response = qa_chain.run(query)
        st.write("📑 Answer:")
        st.success(response)
