import streamlit as st
from pathlib import Path
import tempfile
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# Modern theme configuration
st.set_page_config(
    page_title="🐋 DeepSeek Document Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced modern theme with neural network inspired styling
st.markdown("""
<style>
    /* Global Theme */
    .main {
        background: linear-gradient(135deg, #0a192f 0%, #112240 100%);
        color: #e0e0e0;
    }
    
    /* Neural Network Animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0a192f 0%, #112240 100%);
        border-right: 1px solid #233554;
    }
    .css-1d391kg {
        background: linear-gradient(135deg, #0a192f 0%, #112240 100%);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #3a1c71 0%, #d76d77 50%, #ffaf7b 100%);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(58, 28, 113, 0.3);
        animation: pulse 1.5s infinite;
    }
    
    /* Updated Groq API Input */
    .stTextInput[data-testid="stTextInput"] input {
        background: rgba(30, 42, 58, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(58, 28, 113, 0.3) !important;
        border-radius: 8px;
        padding: 0.8rem;
    }
    
    /* Updated Model Selection Dropdown */
    .stSelectbox[data-testid="stSelectbox"] > div > div {
        background: rgba(30, 42, 58, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(58, 28, 113, 0.3) !important;
        border-radius: 8px;
    }
    
    /* Updated File Uploader */
    .uploadedFile {
        background: rgba(30, 42, 58, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(58, 28, 113, 0.3) !important;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Updated Drag and Drop Area */
    .css-1mvd0qx {
        background: rgba(30, 42, 58, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 2px dashed rgba(58, 28, 113, 0.3) !important;
        color: #e0e0e0 !important;
        max-height: 150px !important;
        min-height: 100px !important;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background-color: #1e2a3a;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #233554;
    }
    
    /* Welcome Card */
    .welcome-card {
        background: linear-gradient(135deg, #1e2a3a 0%, #233554 100%);
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid #233554;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 2rem 0;
        animation: pulse 3s infinite;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #1e2a3a;
    }
    ::-webkit-scrollbar-thumb {
        background: #233554;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #d76d77;
    }

    /* Chat Input */
    .stChatInput {
        border-radius: 8px;
        background-color: #1e2a3a !important;
        border: 1px solid #233554 !important;
        color: #e0e0e0 !important;
    }

    /* Success/Error Messages */
    .stSuccess, .stError {
        background-color: #1e2a3a !important;
        color: #e0e0e0 !important;
        border-radius: 8px;
    }

    /* Text Color */
    .css-10trblm {
        color: #e0e0e0 !important;
    }
    p {
        color: #e0e0e0 !important;
    }
    
    /* DeepSeek Badge */
    .deepseek-badge {
        background: linear-gradient(45deg, #3a1c71 0%, #d76d77 100%);
        padding: 0.2rem 0.8rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        display: inline-block;
        margin-bottom: 1rem;
    }

    /* Sidebar Headers */
    .sidebar-header {
        color: #e0e0e0;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #233554;
    }
</style>
""", unsafe_allow_html=True)

class DeepSeekDocumentChat:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.vector_store = None
        self.chain = None
        self.memory = ConversationBufferMemory()
        
    def _setup_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _setup_llm(self):
        return ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.6,
            max_tokens=4096,
            streaming=True
        )
    
    def process_document(self, file, temp_dir: Path) -> Optional[List[str]]:
        file_path = temp_dir / file.name
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
            
        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader
        }
        
        file_ext = Path(file.name).suffix.lower()
        if file_ext not in loaders:
            st.error(f"Unsupported file format: {file_ext}")
            return None
            
        try:
            loader = loaders[file_ext](str(file_path))
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            return splitter.split_documents(docs)
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return None

    def initialize_system(self, files) -> bool:
        with tempfile.TemporaryDirectory() as temp_dir:
            docs = []
            for file in files:
                if processed_docs := self.process_document(file, Path(temp_dir)):
                    docs.extend(processed_docs)
                    
            if not docs:
                return False
                
            self.vector_store = FAISS.from_documents(docs, self._setup_embeddings())
            self._setup_chain()
            return True
            
    def _setup_chain(self):
        prompt = ChatPromptTemplate.from_template("""
        Expert AI Assistant powered by DeepSeek's advanced cognitive capabilities. Provide comprehensive, nuanced responses based on the context.

        Previous conversation:
        {history}

        Context:
        {context}

        Question: {question}

        Analyze deeply and respond with specific references when possible. If information isn't in the context, clearly state so.
        """)
        
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "lambda_mult": 0.6}
        )
        
        self.chain = {
            "context": retriever,
            "question": RunnablePassthrough(),
            "history": lambda _: self.memory.load_memory_variables({})["history"]
        } | prompt | self._setup_llm()

    async def aquery(self, question: str):
        if not self.chain:
            yield "Please upload documents first."
            return
        try:
            async for chunk in self.chain.astream(question):
                yield chunk.content
        except Exception as e:
            yield f"Error: {str(e)}"

def main():
    st.markdown("""
        <h1 style='text-align: center; color: #e0e0e0;'>
            🐋 DeepSeek Document Analysis
        </h1>
        <div style='text-align: center;'>
            <span class='deepseek-badge'>Powered by Advanced Neural Processing</span>
        </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<p class="sidebar-header">🔧 Configuration</p>', unsafe_allow_html=True)
        
        api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            help="Enter your Groq API key"
        )
        
        model_name = st.selectbox(
            "🧬 Model Selection",
            ["deepseek-r1-distill-llama-70b", "mixtral-8x7b-32768", "llama2-70b-4096"],
            help="Choose your preferred model for analysis"
        )
        
        st.markdown('<p class="sidebar-header">📁 Documents</p>', unsafe_allow_html=True)
        
        files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls']
        )
        
        if files and api_key:
            if st.button("🐋 Initialize Processing", key="process"):
                with st.spinner("🔄 Processing documents..."):
                    chat_system = DeepSeekDocumentChat(api_key, model_name)
                    if chat_system.initialize_system(files):
                        st.session_state.chat_system = chat_system
                        st.session_state.messages = []
                        st.success("✨ Processing complete!")
                    else:
                        st.error("❌ Processing failed")

    if "chat_system" in st.session_state:
        for msg in st.session_state.get('messages', []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        if question := st.chat_input("🐋 Ask about your documents..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Use asyncio to handle streaming
                import asyncio
                async def process_stream():
                    async for chunk in st.session_state.chat_system.aquery(question):
                        nonlocal full_response
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                asyncio.run(process_stream())
    else:
        st.markdown("""
            <div class='welcome-card'>
                <h2 style='text-align: center; color: #e0e0e0; margin-bottom: 1rem;'>
                    🐋 Welcome to DeepSeek Document Analysis
                </h2>
                <p style='text-align: center; color: #e0e0e0;'>
                    Upload your documents and provide your API key to begin the analysis.
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    load_dotenv()
    main()

