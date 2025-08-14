import os
from typing import Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
# from dotenv import load_dotenv

from exceptions import ConfigurationException

# load_dotenv()

# Set Hugging Face cache directory
os.environ["HF_HOME"] = "D:/HF_model"

# Disable LangChain telemetry/analytics
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

# Disable PostHog analytics
os.environ["POSTHOG_HOST"] = ""
os.environ["POSTHOG_PROJECT_API_KEY"] = ""

# Disable LangSmith
os.environ["LANGSMITH_TRACING"] = "false"

@dataclass
class Config:
    """Centralized configuration for the RAG system"""
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = field(default_factory=lambda: os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'))
    
    # Model Configuration
    LLM_MODEL: str = "deepseek-r1:8b"
    GRAPH_LLM_MODEL: str = "deepseek-r1:8b"
    EMBEDDING_MODEL: str = "D:/RAG/qwen-embedding" 
    RERANKER_MODEL: str = "D:/RAG/qwen-reranker"
    ENABLE_GRAPH_PROCESSING: bool = True
    GRAPH_STORE_DIRECTORY: str = 'data/graphstore'
    LLM_TEMPERATURE: float = 0.4

    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_DOCUMENT_SIZE: int = 10_000_000
    
    # Retrieval Configuration
    SEARCH_K: int = 50
    RERANK_K: int = 10
    
    # Hybrid Retrieval Weights
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3
    
    # Vector Store Configuration
    PERSIST_DIRECTORY: str = 'data/vectorstore'
    COLLECTION_NAME: str = 'legal_documents'
    
    # File Processing
    SUPPORTED_EXTENSIONS: tuple = ('.pdf', '.txt', '.docx', '.md')
    TEMP_DIR: str = 'temp'
    
    def __post_init__(self):
        """Initialize computed fields and validate configuration"""
        self._ensure_directories_exist()
        self._validate_ollama_connection()
        self._validate_local_models()
    
    def _ensure_directories_exist(self):
        directories = [
            self.PERSIST_DIRECTORY,
            self.TEMP_DIR,
            self.GRAPH_STORE_DIRECTORY
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _validate_ollama_connection(self):
        """Validate Ollama connection and model availability"""
        try:
            import requests
            response = requests.get(f"{self.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConfigurationException(f"Cannot connect to Ollama at {self.OLLAMA_BASE_URL}")
            
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.LLM_MODEL not in model_names:
                print(f"Warning: Model {self.LLM_MODEL} not found in Ollama. Available models: {model_names}")
                print(f"Please run: ollama pull {self.LLM_MODEL}")
                
        except requests.exceptions.RequestException:
            print(f"Warning: Cannot connect to Ollama at {self.OLLAMA_BASE_URL}. Make sure Ollama is running.")
        except ImportError:
            print("Warning: requests library not available for Ollama validation")
    
    def _validate_local_models(self):
        """Validate local model paths exist"""
        if not Path(self.EMBEDDING_MODEL).exists():
            print(f"Warning: Embedding model not found at {self.EMBEDDING_MODEL}")
        
        if not Path(self.RERANKER_MODEL).exists():
            print(f"Warning: Reranker model not found at {self.RERANKER_MODEL}")
   
    
    # Prompt Templates
    QA_TEMPLATE: str = """Anda adalah asisten AI yang ahli dalam hukum dan peraturan Otoritas Jasa Keuangan (OJK). 
        Gunakan konteks yang diberikan untuk menjawab pertanyaan dengan akurat dan komprehensif.

        Aturan menjawab:
        1. Jika tidak mengetahui jawabannya, katakan "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan tersebut berdasarkan dokumen yang tersedia."
        2. Selalu sebutkan dasar hukum atau peraturan yang relevan jika ada
        3. Jawab dalam bahasa Indonesia yang formal dan mudah dipahami
        4. Jika ada nomor pasal atau ayat yang spesifik, sebutkan dengan jelas
        5. Berikan penjelasan yang praktis dan aplikatif
        6. Jika informasi dari multiple sources, jelaskan dengan jelas sumber masing-masing
        7. WAJIB menyebutkan sumber dokumen dan halaman jika tersedia dalam metadata

        Format konteks mencakup:
        - [SUMBER]: Nama file/dokumen
        - [HALAMAN]: Nomor halaman
        - [BAGIAN]: Struktur hierarkis (Pasal, Ayat, dll)
        - [TANGGAL]: Kapan dokumen diproses
        - [KONTEN]: Isi dokumen

        Konteks: {context}

        Pertanyaan: {question}

        Jawaban:"""

    QUERY_ENHANCEMENT_TEMPLATE: str = """Anda adalah ahli dalam reformulasi query untuk sistem pencarian dokumen hukum.
        Berdasarkan riwayat percakapan dan pertanyaan terbaru, reformulasikan pertanyaan agar lebih spesifik dan mudah dicari dalam dokumen hukum OJK.

        Riwayat percakapan sebelumnya:
        {chat_history}

        Pertanyaan terbaru: {query}

        Berikan reformulasi yang:
        1. Mempertimbangkan konteks dari percakapan sebelumnya
        2. Lebih spesifik dan teknis berdasarkan konteks yang sudah ada
        3. Menggunakan terminologi hukum yang tepat
        4. Fokus pada aspek yang dapat ditemukan dalam dokumen
        5. Tidak mengubah makna pertanyaan asli
        6. Menghubungkan dengan topik yang sudah dibahas sebelumnya jika relevan

        Pertanyaan yang diperbaiki:"""

    GRAPH_QA_TEMPLATE: str = """Anda adalah asisten AI yang ahli dalam analisis hubungan dan struktur dokumen hukum OJK.
        Gunakan informasi graph dan koneksi entitas yang diberikan untuk menjawab pertanyaan.

        Informasi Graph: {graph_context}

        Pertanyaan: {question}

        Berikan jawaban yang:
        1. Memanfaatkan hubungan antar entitas dalam graph
        2. Menjelaskan koneksi dan relasi yang relevan
        3. Menggunakan struktur hierarkis dokumen hukum
        4. Menyebutkan entitas-entitas kunci yang terlibat

        Jawaban:"""