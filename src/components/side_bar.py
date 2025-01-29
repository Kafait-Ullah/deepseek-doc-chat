import streamlit as st

from app import DeepSeekDocumentChat

def render_sidebar():
    """
    Render the complete sidebar with all functionality and styling.
    Returns the configuration values needed by the main application.
    """
    
    # Add sidebar-specific styling
    st.markdown("""
    <style>
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

    with st.sidebar:
        # Configuration Section
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
        
        # Documents Section
        st.markdown('<p class="sidebar-header">📁 Documents</p>', unsafe_allow_html=True)
        
        files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls']
        )
        
        # Process Button
        should_process = False
        if files and api_key:
            if st.button("🐋 Initialize Processing", key="process"):
                with st.spinner("🔄 Processing documents..."):
                    chat_system = DeepSeekDocumentChat(api_key, model_name)
                    if chat_system.initialize_system(files):
                        st.session_state.chat_system = chat_system
                        st.session_state.messages = []
                        st.success("✨ Processing complete!")
                        should_process = True
                    else:
                        st.error("❌ Processing failed")

        return api_key, model_name, files, should_process

# Usage in main.py:
def main():
    # Get configuration from sidebar
    api_key, model_name, files, should_process = render_sidebar()
    
    # Rest of your main code...
