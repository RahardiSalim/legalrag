import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from langchain_google_genai import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
load_dotenv()


@dataclass
class Config:
    """Centralized configuration for the RAG system"""
    
    # API Configuration
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv('GEMINI_API_KEY', ''))
    
    # Model Configuration
    LLM_MODEL: str = "gemini-2.0-flash-exp"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    LLM_TEMPERATURE: float = 0.4
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_DOCUMENT_SIZE: int = 10_000_000  # 10MB limit
    
    # Retrieval Configuration
    SEARCH_K: int = 20
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
    
    # Safety Settings
    SAFETY_SETTINGS: Dict[Any, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed fields and validate configuration"""
        # Ensure directories exist
        Path(self.PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
        Path(self.TEMP_DIR).mkdir(parents=True, exist_ok=True)
        
        # Set safety settings
        if not self.SAFETY_SETTINGS:
            self.SAFETY_SETTINGS = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        
        # Validate API key
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required. Set it as environment variable or in config.")
    
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