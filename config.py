from typing import Dict, Any
from dataclasses import dataclass
from langchain_google_genai import HarmCategory, HarmBlockThreshold


@dataclass
class Config:
    """Centralized configuration for the RAG system"""
    
    # API Configuration
    GEMINI_API_KEY: str = 'YOUR_GEMINI_API_KEY_HERE'
    
    # Model Configuration
    LLM_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    LLM_TEMPERATURE: float = 0.7
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    
    # Retrieval Configuration
    SEARCH_K: int = 20
    FINAL_K: int = 10
    
    # Hybrid Retrieval Weights
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3
    
    # Vector Store Configuration
    PERSIST_DIRECTORY: str = 'docs/chroma/'
    
    # Safety Settings
    SAFETY_SETTINGS: Dict[Any, Any] = None
    
    def __post_init__(self):
        if self.SAFETY_SETTINGS is None:
            self.SAFETY_SETTINGS = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
    
    # Prompt Template for Legal RAG
    QA_TEMPLATE: str = """Anda adalah asisten AI yang ahli dalam hukum dan peraturan Otoritas Jasa Keuangan (OJK). 
    Gunakan konteks yang diberikan untuk menjawab pertanyaan dengan akurat dan komprehensif.
    
    Aturan menjawab:
    1. Jika tidak mengetahui jawabannya, katakan "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan tersebut berdasarkan dokumen yang tersedia."
    2. Selalu sebutkan dasar hukum atau peraturan yang relevan jika ada
    3. Jawab dalam bahasa Indonesia yang formal dan mudah dipahami
    4. Jika ada nomor pasal atau ayat yang spesifik, sebutkan dengan jelas
    5. Berikan penjelasan yang praktis dan aplikatif
    
    Konteks: {context}
    
    Pertanyaan: {question}
    
    Jawaban:"""