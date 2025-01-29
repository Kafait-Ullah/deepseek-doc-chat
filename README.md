# 🐋 DeepSeek Document Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-Powered-brightgreen)](https://deepseek.com/)

## 🎯 Overview

DeepSeek Document Analysis is a powerful document analysis tool that leverages DeepSeek's advanced language models through Groq's infrastructure. This application enables users to analyze multiple document formats and engage in intelligent conversations about their content using state-of-the-art AI technology.

## ✨ Features

- 🚀 Powered by DeepSeek's cutting-edge language models
- 📄 Support for multiple file formats (PDF, DOCX, TXT, CSV, XLSX, XLS)
- 💬 Interactive chat interface for document analysis
- 🎨 Modern, neural network-inspired UI
- 🔄 Real-time document processing
- 🧠 Advanced context-aware responses
- 🔒 Secure API key handling

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Kafait-Ullah/deepseek-document-analysis.git

# Navigate to project directory
cd deepseek-document-analysis

# Install requirements
pip install -r requirements.txt
```

## 📋 Requirements

```
streamlit>=1.28.0
langchain>=0.1.0
langchain-community>=0.0.10
langchain-groq>=0.0.3
langchain-huggingface>=0.0.2
faiss-cpu>=1.7.4
python-dotenv>=1.0.0
sentence-transformers>=2.2.2
pypdf>=3.17.1
docx2txt>=0.8
pandas>=2.0.0
openpyxl>=3.1.2
```

## 🚀 Getting Started

### 1. Get Your Groq API Key

1. Visit [Groq Console](https://console.groq.com/playground?model=deepseek-r1-distill-llama-70b)
2. Sign up or log in to your account
3. Navigate to API section
4. Generate your API key
5. Copy the API key for use in the application

### 2. Launch the Application

```bash
streamlit run app.py
```

### 3. Using the Application

1. **Configure Settings:**
   - Paste your Groq API key in the sidebar
   - Select DeepSeek model (recommended: deepseek-r1-distill-llama-70b)

2. **Upload Documents:**
   - Click the upload button in the sidebar
   - Select one or multiple supported documents
   - Supported formats: PDF, DOCX, TXT, CSV, XLSX, XLS

3. **Initialize Processing:**
   - Click "Initialize Processing" button
   - Wait for document processing to complete

4. **Start Analysis:**
   - Use the chat interface to ask questions about your documents
   - Receive AI-powered responses based on document content

## 🌟 Why DeepSeek?

DeepSeek's language models represent the cutting edge in AI technology:

- **Advanced Understanding:** Superior comprehension of complex documents
- **Contextual Awareness:** Maintains context across multiple interactions
- **Efficient Processing:** Fast and accurate responses through Groq's infrastructure
- **Scalability:** Handles multiple documents and large content volumes effectively

## 🔐 Security

- API keys are handled securely and never stored
- Documents are processed locally
- Temporary files are automatically cleaned up
- No data is permanently stored on servers

## 📄 License

MIT License

Copyright (c) 2024 Kafait-Ullah

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## 👨‍💻 Author

**Kafait-Ullah**
- GitHub: [@Kafait-Ullah](https://github.com/Kafait-Ullah)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/Kafait-Ullah/deepseek-document-analysis/issues).

## ⭐ Show your support

Give a ⭐️ if this project helped you!































Ah yes, let me correct that with a more appropriate open-source focused post:

🔧 New Open-Source Project: Document Analysis System with DeepSeek Integration

Built a straightforward RAG implementation leveraging DeepSeek models via Groq. Wanted to share this with the community for those interested in exploring DeepSeek's capabilities in document analysis applications.

🔍 Core Features:
• Document vectorization and retrieval
• Clean UI for document interaction
• Multi-format support
• DeepSeek model integration

⚡️ Tech Implementation:
• DeepSeek via Groq
• LangChain
• FAISS
• Streamlit

📘 Open-source and available on GitHub: [Link]

Feel free to explore, contribute, or adapt for your projects. Looking forward to community feedback and potential improvements.

#OpenSource #AI #DeepSeek #MachineLearning #Python

This better captures the open-source nature of your project? Let me know if you'd like any adjustments to the tone or content.