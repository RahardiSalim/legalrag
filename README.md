# LegalRAG 🏛️⚖️

> **Intelligent Legal Document Assistant for Indonesian Financial Services Authority (OJK) Regulations**

A sophisticated Retrieval-Augmented Generation (RAG) system built with FastAPI and Streamlit that helps legal professionals, compliance officers, and financial institutions navigate complex OJK regulations and Indonesian financial law.

## Features

- **Multi-format Document Processing**: Upload PDF, TXT, and ZIP files
- **Hybrid Search**: Combines vector similarity and BM25 keyword matching
- **AI-Powered Q&A**: Natural language queries about legal documents
- **Enhanced Query Processing**: Intelligent query reformulation for better results
- **Source Transparency**: View exact document chunks used for answers
- **Conversational Interface**: Maintains context across multiple questions
- **Dual Interface**: REST API and interactive Streamlit web app
- **Real-time Processing**: Fast document indexing and retrieval

## Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key
- 4GB+ RAM (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/legalrag-ojk.git
   cd legalrag-ojk
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key**
   ```bash
   # Edit config.py and add your Gemini API key
   GEMINI_API_KEY = 'your_api_key_here'
   ```

4. **Start the API server**
   ```bash
   python main.py
   ```

5. **Launch the web interface**
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

### Web Interface

1. **Upload Documents**: Navigate to the sidebar and upload your legal documents (PDF/TXT/ZIP)
2. **Ask Questions**: Type your legal questions in Indonesian or English
3. **Enhanced Search**: Toggle enhanced query processing for complex questions
4. **View Sources**: Check which document sections were used for each answer

### API Endpoints

```python
# Upload documents
POST /upload

# Chat with the system
POST /chat
{
  "question": "Apa itu peraturan OJK tentang fintech?",
  "use_enhanced_query": false
}

# Get chat history
GET /history

# Clear conversation
POST /clear-history

# View last retrieved chunks
GET /chunks

# Health check
GET /health
```

## Architecture
![Architecture](assets/architecture.png)

## Configuration

Edit `config.py` to customize:

- **Model Settings**: LLM model, temperature, embedding model
- **Search Parameters**: Chunk size, overlap, retrieval count
- **Hybrid Weights**: Balance between vector and keyword search
- **Safety Settings**: Content filtering options

## License

This project is licensed under the MIT License

## Acknowledgments

- **LangChain** for the RAG framework
- **Google Gemini** for language model capabilities
- **ChromaDB** for vector storage
- **Streamlit** for the web interface
- **FastAPI** for the REST API

---
© Rahardi Salim
**⚖️ Built for Indonesian Legal Professionals | 🏛️ Empowering Financial Compliance**