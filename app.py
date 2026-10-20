import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

st.set_page_config(page_title="RAG PDF Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG PDF Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

@st.cache_resource
def load_models():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3", temperature=0)
    return embeddings, llm

embeddings, llm = load_models()

with st.sidebar:
    st.header("Document Setup")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Chunking & Embedding Document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)

            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            
            os.remove(tmp_path)
            st.success(f"Indexed {len(chunks)} text chunks!")
            st.session_state.messages = []

if st.session_state.vector_store is None:
    st.info("👈 Upload a PDF in the sidebar to begin chatting.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document context..."):
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(prompt)
                
                context = "\n\n---\n\n".join([d.page_content for d in docs])
                
                full_prompt = f"""
You are a helpful study assistant. Use the provided document context below to answer the user's question clearly and accurately.

Context:
{context}

Question: {prompt}

Answer:
"""
                response = llm.invoke(full_prompt)
                answer = response.content
                
                st.markdown(answer)
                
                with st.expander("View Source Chunks"):
                    for i, doc in enumerate(docs):
                        page_num = doc.metadata.get("page", 0) + 1
                        st.caption(f"**Chunk {i+1} (Page {page_num}):** {doc.page_content[:150]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})