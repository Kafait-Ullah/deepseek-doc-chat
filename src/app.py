# app.py
import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURE LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# WORKAROUND FOR STREAMLIT + ASYNCIO
# ─────────────────────────────────────────────────────────────────────────────
nest_asyncio.apply()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐋 DeepSeek Document Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* ─── CONTAINER & BACKGROUND ───────────────────────────────────────────── */
.main { background: linear-gradient(135deg, #0a192f 0%, #112240 100%); color: #e0e0e0; }
section[data-testid="stSidebar"] { background: linear-gradient(135deg, #0a192f, #112240); }

/* ─── BUTTONS ────────────────────────────────────────────────────────── */
.stButton>button {
    background: linear-gradient(45deg, #3a1c71, #d76d77, #ffaf7b);
    color: #fff; border-radius: 8px; padding: .6rem 1.2rem;
    border: none; font-weight: 600; transition: 0.3s; width: 100%; margin-top: 1rem;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(58,28,113,0.3); }

/* ─── TEXT INPUTS ────────────────────────────────────────────────────── */
/* unify icon/bg + focused/unfocused state */
[data-testid="stTextInput"] > div, 
[data-testid="stTextInput"] input, 
[data-testid="stTextInput"] svg {
    background: rgba(30,42,58,0.8) !important;
    color: #e0e0e0 !important;
    border: 1px solid rgba(58,28,113,0.3) !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(58,28,113,0.5) !important;
}

/* ─── SELECTBOX ─────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(30,42,58,0.8) !important;
    color: #e0e0e0 !important;
    border: 1px solid rgba(58,28,113,0.3) !important;
    border-radius: 8px !important;
}

/* ─── FILE UPLOADER ─────────────────────────────────────────────────── */
.css-1mvd0qx, .uploadedFile {
    background: rgba(30,42,58,0.8) !important;
    border: 2px dashed rgba(58,28,113,0.3) !important;
    border-radius: 8px !important;
    max-height: 200px !important;
    overflow-y: auto !important;
    padding: 1rem !important;
    color: #e0e0e0 !important;
}

/* ─── CHAT BOX ─────────────────────────────────────────────────────── */
.stChatInput textarea {
    background: #1e2a3a !important;
    color: #e0e0e0 !important;
    border: 1px solid #233554 !important;
    border-radius: 8px !important;
    padding: .8rem !important;
}
.stChatInput textarea:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(58,28,113,0.5) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# DEEPSEEK DOCUMENT CHAT CLASS
# ─────────────────────────────────────────────────────────────────────────────
class DeepSeekDocumentChat:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.vector_store = None
        self.chain = None
        # use return_messages=True to avoid deprecation warning
        self.memory = ConversationBufferMemory(return_messages=True)

    def _setup_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _setup_llm(self):
        return ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.6,
            max_tokens=4096,
            streaming=True,
        )

    def process_document(self, file, temp_dir: Path) -> Optional[List]:
        file_path = temp_dir / file.name
        file_ext = file_path.suffix.lower()
        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
            ".csv": CSVLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xls": UnstructuredExcelLoader,
        }
        if file_ext not in loaders:
            st.error(f"Unsupported format: {file_ext}")
            return None

        try:
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            loader = loaders[file_ext](str(file_path))
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            return splitter.split_documents(docs)
        except Exception as e:
            logger.exception("Error loading document %s", file.name)
            st.error(f"Failed to load {file.name}: {e}")
            return None

    def initialize_system(self, files) -> bool:
        try:
            all_chunks = []
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                for f in files:
                    chunks = self.process_document(f, tmp_path)
                    if chunks:
                        all_chunks.extend(chunks)
            if not all_chunks:
                return False

            self.vector_store = FAISS.from_documents(all_chunks, self._setup_embeddings())
            self._setup_chain()
            return True
        except Exception as e:
            logger.exception("Initialization failed")
            st.error(f"Initialization error: {e}")
            return False

    def _setup_chain(self):
        prompt = ChatPromptTemplate.from_template(
            """
You are DeepSeek: an expert AI assistant with context from the user’s documents.

History:
{history}

Context:
{context}

User Question: {question}

Provide a detailed answer, citing context when possible, or admit if the info isn’t available.
"""
        )
        retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.6})
        # build a runnable chain
        self.chain = {
            "context": retriever,
            "question": RunnablePassthrough(),
            "history": lambda _: self.memory.load_memory_variables({})["history"],
        } | prompt | self._setup_llm()

    async def aquery(self, question: str):
        if not self.chain:
            yield "Please upload & initialize first."
            return
        try:
            async for chunk in self.chain.astream(question):
                yield chunk.content
        except Exception as e:
            logger.exception("Error during query stream")
            yield f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    load_dotenv()

    st.markdown(
        "<h1 style='text-align:center;color:#e0e0e0;'>🐋 DeepSeek Document Analysis</h1>",
        unsafe_allow_html=True,
    )
    # st.sidebar.header("🔧 Configuration")
    api_key = st.sidebar.text_input("🔑 Groq API Key", type="password")
    model_name = st.sidebar.selectbox(
        "🧬 Model Selection",
        ["deepseek-r1-distill-llama-70b", "mixtral-8x7b-32768", "llama2-70b-4096"],
    )

    st.sidebar.header("📁 Documents")
    files = st.sidebar.file_uploader(
        "Upload (pdf, docx, txt, csv, xlsx, xls)",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
    )

    # initialize
    if files and api_key:
        if st.sidebar.button("🐋 Initialize Processing"):
            st.sidebar.info("🌀 Processing...")
            chat = DeepSeekDocumentChat(api_key, model_name)
            if chat.initialize_system(files):
                st.session_state.chat = chat
                st.session_state.history = []
                st.sidebar.success("✅ Ready! Ask your questions below.")
            else:
                st.sidebar.error("❌ Initialization failed.")

    # chat interface
    if "chat" in st.session_state:
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("🐋 Ask about your documents…")
        if question:
            st.session_state.history.append({"role": "user", "content": question})
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full = ""
                import asyncio

                async def _stream():
                    nonlocal full
                    async for token in st.session_state.chat.aquery(question):
                        full += token
                        placeholder.markdown(full + "▌")
                    placeholder.markdown(full)
                    st.session_state.history.append({"role": "assistant", "content": full})

                asyncio.get_event_loop().run_until_complete(_stream())
    else:
        st.markdown(
            """
<div style='margin:2rem;padding:2rem;background:#233554;border-radius:12px;'>
<h2 style='color:#e0e0e0;text-align:center;'>🐋 Welcome to DeepSeek</h2>
<p style='color:#e0e0e0;text-align:center;'>Upload docs & your API key to begin.</p>
</div>
""",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Unhandled exception in Streamlit app")
        st.error(f"Fatal error: {e}")
