import os
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Disable all analytics and telemetry FIRST
os.environ.update({
    # LangChain analytics
    "LANGCHAIN_TRACING_V2": "false",
    "LANGCHAIN_ENDPOINT": "",
    "LANGCHAIN_API_KEY": "",
    "LANGCHAIN_PROJECT": "",
    
    # PostHog analytics
    "POSTHOG_HOST": "",
    "POSTHOG_PROJECT_API_KEY": "",
    "POSTHOG_DISABLED": "true",
    "POSTHOG_CAPTURE": "false",
    
    # LangSmith
    "LANGSMITH_TRACING": "false",
    "LANGSMITH_API_KEY": "",
    
    # Chroma and other analytics
    "CHROMA_TELEMETRY_ENABLED": "false",
    "ANONYMIZED_TELEMETRY": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "DO_NOT_TRACK": "1",
    "TELEMETRY_DISABLED": "true",
    
    # Hugging Face cache
    "HF_HOME": "D:/HF_model"
})

from core.exceptions import ConfigurationException


@dataclass
class ModelConfig:
    """AI Model Configuration"""
    # Ollama Configuration
    ollama_base_url: str = field(default_factory=lambda: os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'))
    
    # Model Paths and Names
    llm_model: str = "qwen3:4b"
    graph_llm_model: str = "qwen3:4b"
    embedding_model: str = "D:/RAG/qwen-embedding"
    reranker_model: str = "D:/RAG/qwen-reranker"
    
    # Model Parameters
    llm_temperature: float = 0.4
    llm_repeat_penalty: float = 1.1
    llm_top_k: int = 40
    llm_top_p: float = 0.9
    llm_num_ctx: int = 8192
    
    # Device Configuration
    device: str = "cpu"
    embedding_device: str = "cpu"
    reranker_device: str = "cpu"


@dataclass
class DocumentConfig:
    """Document Processing Configuration"""
    # File Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_document_size: int = 10_000_000
    supported_extensions: tuple = ('.pdf', '.txt', '.docx', '.md')
    
    # Text Processing
    min_chunk_length: int = 50
    max_content_truncate: int = 1000
    max_combined_content_length: int = 6000
    max_docs_for_graph: int = 5
    doc_truncate_length: int = 1200
    
    # Legal Document Detection
    legal_indicators: list = field(default_factory=lambda: [
        'pasal', 'bab', 'bagian', 'ayat', 'peraturan', 'undang-undang',
        'keputusan', 'surat edaran', 'ojk', 'otoritas jasa keuangan',
        'peraturan otoritas jasa keuangan', 'pojk'
    ])
    legal_detection_threshold: int = 3


@dataclass
class RetrievalConfig:
    """Information Retrieval Configuration"""
    # Search Parameters
    search_k: int = 50
    rerank_k: int = 10
    max_results_default: int = 5
    max_results_limit: int = 20
    
    # Hybrid Retrieval Weights
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    
    # Similarity Thresholds
    similarity_threshold: float = 0.3
    graph_entry_points_limit: int = 5
    semantic_similarity_threshold: float = 0.2


@dataclass
class GraphConfig:
    """Knowledge Graph Configuration"""
    # Graph Processing
    enable_graph_processing: bool = True
    graph_store_directory: str = 'data/graphstore'
    
    # Graph Search Parameters
    max_graph_nodes: int = 20
    graph_search_depth: int = 2
    graph_neighbor_score_decay: float = 0.7
    
    # Graph Batch Processing
    graph_batch_size: int = 10
    max_graph_text_length: int = 1200
    
    # Graph Visualization
    graph_visualization_height: str = "800px"
    graph_visualization_width: str = "100%"
    graph_node_size: int = 25
    graph_edge_width: int = 2


@dataclass
class StorageConfig:
    """Storage Configuration"""
    # Vector Store
    persist_directory: str = 'data/vectorstore'
    collection_name: str = 'legal_documents'
    
    # File Storage
    temp_dir: str = 'temp'
    
    # Feedback Storage
    feedback_storage_path: str = "data/feedback.jsonl"


@dataclass
class MemoryConfig:
    """Memory and Caching Configuration"""
    # Chat Memory
    conversation_window_size: int = 5
    max_chat_history_display: int = 6
    
    # Feedback Cache
    feedback_cache_ttl: int = 300  # 5 minutes
    
    # Content Limits
    max_response_content_length: int = 300
    max_source_name_length: int = 60
    source_name_truncate_suffix: str = "..."


@dataclass
class FeedbackConfig:
    """Feedback Learning Configuration"""
    # Learning Parameters
    min_similarity_threshold: float = 0.2
    max_adjustment: float = 0.3
    document_relevance_threshold: float = 0.1
    
    # Feedback Scoring
    score_range_min: int = 1
    score_range_max: int = 5
    max_comment_length: int = 1000
    
    # Query Similarity
    jaccard_similarity_enabled: bool = True


@dataclass
class SystemConfig:
    """System-wide Configuration"""
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    
    # Logging
    log_level: str = "INFO"
    
    # Default Language
    default_language: str = "id"  # Indonesian


@dataclass
class Settings:
    """Main Settings Class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """Initialize and validate settings"""
        self._ensure_directories_exist()
        self._validate_configuration()
    
    def _ensure_directories_exist(self):
        """Create necessary directories"""
        directories = [
            self.storage.persist_directory,
            self.storage.temp_dir,
            self.graph.graph_store_directory,
            Path(self.storage.feedback_storage_path).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _validate_configuration(self):
        """Validate configuration settings"""
        try:
            # Validate paths
            if not Path(self.model.embedding_model).exists():
                print(f"Warning: Embedding model not found at {self.model.embedding_model}")
            
            if not Path(self.model.reranker_model).exists():
                print(f"Warning: Reranker model not found at {self.model.reranker_model}")
            
            # Validate Ollama connection
            self._validate_ollama_connection()
            
        except Exception as e:
            print(f"Configuration validation warning: {e}")
    
    def _validate_ollama_connection(self):
        """Validate Ollama connection"""
        try:
            import requests
            response = requests.get(f"{self.model.ollama_base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConfigurationException(f"Cannot connect to Ollama at {self.model.ollama_base_url}")
            
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model.llm_model not in model_names:
                print(f"Warning: Model {self.model.llm_model} not found in Ollama. Available models: {model_names}")
                
        except requests.exceptions.RequestException:
            print(f"Warning: Cannot connect to Ollama at {self.model.ollama_base_url}. Make sure Ollama is running.")
        except ImportError:
            print("Warning: requests library not available for Ollama validation")
    
    @property
    def prompt_templates(self) -> Dict[str, str]:
        """Get prompt templates"""
        return {
            "qa_template": """Anda adalah asisten AI yang ahli dalam hukum dan peraturan Otoritas Jasa Keuangan (OJK). 
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

            Jawaban:""",
            
            "query_enhancement_template": """Anda adalah ahli dalam reformulasi query untuk sistem pencarian dokumen hukum.
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

            Pertanyaan yang diperbaiki:""",
            
            "graph_qa_template": """Anda adalah asisten AI yang ahli dalam analisis hubungan dan struktur dokumen hukum OJK.
            Gunakan informasi graph dan koneksi entitas yang diberikan untuk menjawab pertanyaan.

            Informasi Graph: {graph_context}

            Pertanyaan: {question}

            Berikan jawaban yang:
            1. Memanfaatkan hubungan antar entitas dalam graph
            2. Menjelaskan koneksi dan relasi yang relevan
            3. Menggunakan struktur hierarkis dokumen hukum
            4. Menyebutkan entitas-entitas kunci yang terlibat

            Jawaban:"""
        }


# Global settings instance
settings = Settings()

# Legacy Config class for backward compatibility
class Config:
    def __init__(self):
        self._settings = settings
    
    # Model Configuration
    @property
    def OLLAMA_BASE_URL(self): return self._settings.model.ollama_base_url
    @property
    def LLM_MODEL(self): return self._settings.model.llm_model
    @property
    def GRAPH_LLM_MODEL(self): return self._settings.model.graph_llm_model
    @property
    def EMBEDDING_MODEL(self): return self._settings.model.embedding_model
    @property
    def RERANKER_MODEL(self): return self._settings.model.reranker_model
    @property
    def LLM_TEMPERATURE(self): return self._settings.model.llm_temperature
    @property
    def ENABLE_GRAPH_PROCESSING(self): return self._settings.graph.enable_graph_processing
    @property
    def GRAPH_STORE_DIRECTORY(self): return self._settings.graph.graph_store_directory
    
    # Document Configuration
    @property
    def CHUNK_SIZE(self): return self._settings.document.chunk_size
    @property
    def CHUNK_OVERLAP(self): return self._settings.document.chunk_overlap
    @property
    def MAX_DOCUMENT_SIZE(self): return self._settings.document.max_document_size
    @property
    def SUPPORTED_EXTENSIONS(self): return self._settings.document.supported_extensions
    
    # Retrieval Configuration
    @property
    def SEARCH_K(self): return self._settings.retrieval.search_k
    @property
    def RERANK_K(self): return self._settings.retrieval.rerank_k
    @property
    def VECTOR_WEIGHT(self): return self._settings.retrieval.vector_weight
    @property
    def BM25_WEIGHT(self): return self._settings.retrieval.bm25_weight
    
    # Storage Configuration
    @property
    def PERSIST_DIRECTORY(self): return self._settings.storage.persist_directory
    @property
    def COLLECTION_NAME(self): return self._settings.storage.collection_name
    @property
    def TEMP_DIR(self): return self._settings.storage.temp_dir
    
    # Prompt Templates
    @property
    def QA_TEMPLATE(self): return self._settings.prompt_templates["qa_template"]
    @property
    def QUERY_ENHANCEMENT_TEMPLATE(self): return self._settings.prompt_templates["query_enhancement_template"]
    @property
    def GRAPH_QA_TEMPLATE(self): return self._settings.prompt_templates["graph_qa_template"]
    
    # Add new property for offline mode detection
    @property
    def OFFLINE_MODE(self):
        try:
            import requests
            response = requests.get(f"{self.OLLAMA_BASE_URL}/api/tags", timeout=2)
            return response.status_code != 200
        except Exception:
            return True