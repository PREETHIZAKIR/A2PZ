import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# === CONFIGURE YOUR GOOGLE API KEY ===
GOOGLE_API_KEY = "AIzaSyDK5hw1kVgVBgJJc64SHH7T9pJOWM2U2lk" # 🔐 Replace with actual key

# === SETUP Gemini LLM and Embeddings ===
llm = ChatGoogleGenerativeAI(model="gemini-2.0 flash", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY)

# === STREAMLIT UI ===
st.set_page_config(page_title="📄 PDF Q&A with RAG", layout="centered")
st.title("🔍 Ask Questions from Your PDF using RAG")

