import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Search Type Enum (matching the API)
class SearchType(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    GRAPH = "graph"

# Configure the page
st.set_page_config(
    page_title="Legal RAG Assistant with Graph Support",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.stAlert > div {
    padding: 0.5rem 1rem;
}
.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 0.5rem;
    color: #1f1f1f; 
    line-height: 1.6;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 5px solid #1e88e5;
}
.assistant-message {
    background-color: #e8f5e9;
    border-left: 5px solid #43a047;
}
.chunk-container {
    border: 1px solid #e0e0e0;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
    background-color: #fafafa;
}
.metric-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.metadata-item {
    background-color: #e0e0e0;
    color: #212121;
    padding: 0.3rem 0.6rem;
    margin: 0.2rem;
    border-radius: 0.3rem;
    display: inline-block;
    font-size: 0.8rem;
}
.score-badge {
    background-color: #e3f2fd;
    color: #1976d2;
    padding: 0.2rem 0.5rem;
    border-radius: 0.3rem;
    font-weight: bold;
    font-size: 0.8rem;
}
.graph-badge {
    background-color: #f3e5f5;
    color: #7b1fa2;
    padding: 0.2rem 0.5rem;
    border-radius: 0.3rem;
    font-weight: bold;
    font-size: 0.8rem;
}
.entity-badge {
    background-color: #fff3e0;
    color: #f57c00;
    padding: 0.2rem 0.5rem;
    border-radius: 0.3rem;
    margin: 0.1rem;
    display: inline-block;
    font-size: 0.75rem;
}
.relationship-badge {
    background-color: #e0f2f1;
    color: #00695c;
    padding: 0.2rem 0.5rem;
    border-radius: 0.3rem;
    margin: 0.1rem;
    display: inline-block;
    font-size: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 600
MAX_RETRIES = 3

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'graph_initialized' not in st.session_state:
        st.session_state.graph_initialized = False
    if 'last_chunks' not in st.session_state:
        st.session_state.last_chunks = []
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = None
    if 'system_stats' not in st.session_state:
        st.session_state.system_stats = {}
    if 'graph_stats' not in st.session_state:
        st.session_state.graph_stats = {}
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False
    if 'search_type' not in st.session_state:
        st.session_state.search_type = SearchType.VECTOR
    if 'last_response' not in st.session_state:
        st.session_state.last_response = {}

# API Helper Functions
class APIClient:
    """Centralized API client with error handling and retries"""
    
    @staticmethod
    def _make_request(method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make API request with retry logic"""
        url = f"{API_BASE_URL}{endpoint}"
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    timeout=REQUEST_TIMEOUT,
                    **kwargs
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 400:
                    error_data = response.json()
                    st.error(f"❌ {error_data.get('detail', 'Bad request')}")
                    return None
                elif response.status_code == 500:
                    error_data = response.json()
                    st.error(f"🔥 Server error: {error_data.get('detail', 'Internal server error')}")
                    return None
                else:
                    st.error(f"❌ Request failed with status {response.status_code}")
                    return None
                    
            except requests.exceptions.Timeout:
                st.error(f"⏱️ Request timeout (attempt {attempt + 1}/{MAX_RETRIES})")
            except requests.exceptions.ConnectionError:
                st.error(f"🔌 Connection error (attempt {attempt + 1}/{MAX_RETRIES})")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Request error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)  # Wait before retry
        
        return None
    
    @staticmethod
    def upload_files(files, enable_graph_processing: bool = True) -> Optional[Dict]:
        """Upload files to the API with graph processing option"""
        files_data = []
        for file in files:
            files_data.append(
                ("files", (file.name, file.getvalue(), file.type))
            )
        
        # Add the graph processing parameter
        data = {"enable_graph_processing": enable_graph_processing}
        
        return APIClient._make_request("POST", "/upload", files=files_data, data=data)
    
    @staticmethod
    def send_chat_message(question: str, search_type: SearchType = SearchType.VECTOR, use_enhanced_query: bool = False) -> Optional[Dict]:
        """Send chat message to the API"""
        payload = {
            "question": question,
            "search_type": search_type.value,
            "use_enhanced_query": use_enhanced_query
        }
        return APIClient._make_request("POST", "/chat", json=payload)
    
    @staticmethod
    def get_chat_history() -> Optional[Dict]:
        """Get chat history from the API"""
        return APIClient._make_request("GET", "/history")
    
    @staticmethod
    def clear_chat_history() -> Optional[Dict]:
        """Clear chat history"""
        return APIClient._make_request("POST", "/clear-history")
    
    @staticmethod
    def get_last_chunks() -> Optional[List]:
        """Get last retrieved chunks"""
        return APIClient._make_request("GET", "/chunks")
    
    @staticmethod
    def check_health() -> Dict:
        """Check API health"""
        result = APIClient._make_request("GET", "/health")
        return result if result else {"status": "error", "system_initialized": False}
    
    @staticmethod
    def get_system_stats() -> Optional[Dict]:
        """Get system statistics"""
        return APIClient._make_request("GET", "/stats")
    
    @staticmethod
    def get_graph_stats() -> Optional[Dict]:
        """Get graph statistics"""
        return APIClient._make_request("GET", "/graph/stats")
    
    @staticmethod
    def create_graph_visualization(filename: str = "graph_visualization.html") -> Optional[Dict]:
        """Create graph visualization via GET request"""
        try:
            return APIClient._make_request("GET", f"/graph/visualize?filename={filename}")
        except Exception as e:
            logger.error(f"Failed to create graph visualization: {e}")
            return None

# UI Components
def render_sidebar():
    """Render the sidebar with upload and settings"""
    with st.sidebar:
        st.header("📁 Upload Dokumen")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload file PDF, TXT, atau ZIP",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'zip'],
            help="Pilih satu atau lebih file untuk diproses"
        )
        
        # Graph processing toggle
        enable_graph_processing = st.checkbox(
            "🕸️ Enable Graph Processing",
            value=True,
            help="Process documents into knowledge graph for advanced search capabilities"
        )
        
        if not enable_graph_processing:
            st.warning("⚠️ Graph features will be disabled. Only vector search will be available.")
        
        # Upload button
        if st.button("📤 Upload dan Proses", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("Memproses dokumen..."):
                    start_time = time.time()
                    result = APIClient.upload_files(uploaded_files, enable_graph_processing)
                    
                    if result and result.get('success'):
                        processing_time = time.time() - start_time
                        
                        # Show success message with details
                        success_msg = f"✅ {result['message']} dalam {processing_time:.2f}s"
                        st.success(success_msg)
                        
                        # Show processing details
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("📄 Files", result.get('file_count', 0))
                        with col2:
                            st.metric("🔗 Chunks", result.get('chunk_count', 0))
                        
                        # Show graph processing results
                        if result.get('graph_processed'):
                            st.success("🕸️ Graph berhasil diproses!")
                            graph_col1, graph_col2 = st.columns(2)
                            with graph_col1:
                                st.metric("Nodes", result.get('graph_nodes', 0))
                            with graph_col2:
                                st.metric("Relations", result.get('graph_relationships', 0))
                            st.session_state.graph_initialized = True
                        elif enable_graph_processing:
                            st.warning("⚠️ Graph processing was enabled but failed")
                        else:
                            st.info("📊 Graph processing was disabled")
                        
                        # Update system state
                        st.session_state.system_initialized = True
                        
                        # Auto refresh
                        time.sleep(1)
                        st.rerun()
                    else:
                        error_msg = result.get('detail', 'Unknown error') if result else 'Connection failed'
                        st.error(f"❌ Upload gagal: {error_msg}")
            else:
                st.warning("⚠️ Pilih file terlebih dahulu!")
        
        # Show upload tips
        with st.expander("💡 Tips Upload"):
            st.write("""
            **Graph Processing:**
            - ✅ **Enabled**: Mendukung Hybrid & Graph search
            - ❌ **Disabled**: Hanya Vector search, lebih cepat
            
            **File Types:**
            - PDF documents
            - Text files (.txt)
            - ZIP archives (akan diekstrak)
            
            **Best Practices:**
            - Upload legal documents untuk hasil terbaik
            - File berukuran wajar (< 10MB per file)
            - Gunakan Graph Processing untuk analisis relasi
            """)
        
        st.markdown("---")
        
        # System Status
        render_system_status()
        
        st.markdown("---")
        
        # Settings
        search_type, use_enhanced_query, max_results = render_settings()
        
        st.markdown("---")
        
        # Graph Section
        render_graph_section()
        
        st.markdown("---")
        
        # Actions
        render_actions()
        
        return search_type, use_enhanced_query, max_results

def render_system_status():
    """Render system status information"""
    st.header("🔧 Status Sistem")
    
    # Get health status
    health = APIClient.check_health()
    
    if health.get('system_initialized'):
        st.success("✅ Sistem Siap")
        st.session_state.system_initialized = True
        
        # Check graph status
        if health.get('graph_store_status') == 'healthy':
            st.success("🕸️ Graph Siap")
            st.session_state.graph_initialized = True
        else:
            st.info("📄 Vector Only")
            st.session_state.graph_initialized = False
        
        # Show additional stats if available
        stats = APIClient.get_system_stats()
        if stats:
            st.session_state.system_stats = stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Dokumen", stats.get('document_count', 0))
            with col2:
                st.metric("🔗 Chunks", stats.get('chunk_count', 0))
            
            # Show available search types
            available_types = stats.get('search_types_available', [])
            if available_types:
                st.caption("🔍 Search types: " + ", ".join([t for t in available_types if t]))
            
            if stats.get('last_upload_time'):
                st.caption(f"📅 Upload terakhir: {stats['last_upload_time']}")
    else:
        st.warning("⚠️ Upload dokumen terlebih dahulu")
        st.session_state.system_initialized = False
        st.session_state.graph_initialized = False
    
    # Health indicators
    with st.expander("🏥 Health Check"):
        st.json(health)

def render_settings():
    """Render settings panel"""
    st.header("⚙️ Pengaturan")
    
    # Search type selection - only show available options
    search_options = ["Vector Search"]
    search_values = [SearchType.VECTOR]
    
    if st.session_state.graph_initialized:
        search_options.extend(["Hybrid Search", "Graph Search"])
        search_values.extend([SearchType.HYBRID, SearchType.GRAPH])
    else:
        st.info("🕸️ Upload documents with Graph Processing enabled to unlock advanced search options")
    
    search_type_idx = st.selectbox(
        "🔍 Tipe Pencarian",
        range(len(search_options)),
        format_func=lambda x: search_options[x],
        index=0,
        help="Pilih metode pencarian berdasarkan ketersediaan data"
    )
    
    search_type = search_values[search_type_idx]
    st.session_state.search_type = search_type
    
    # Show search type explanations
    if search_type == SearchType.VECTOR:
        st.caption("📊 Pencarian similarity berbasis embedding vector")
        if not st.session_state.graph_initialized:
            st.caption("💡 Graph search tidak tersedia - upload dengan Graph Processing untuk fitur lebih advanced")
    elif search_type == SearchType.HYBRID:
        st.caption("🔗 Kombinasi vector search + knowledge graph")
    elif search_type == SearchType.GRAPH:
        st.caption("🕸️ Pencarian berbasis relasi dalam knowledge graph")
    
    # Query enhancement toggle
    use_enhanced_query = st.checkbox(
        "🔍 Gunakan Enhanced Query",
        value=True,
        help="Menggunakan AI untuk memperbaiki pertanyaan Anda berdasarkan konteks percakapan"
    )
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox(
        "🔄 Auto Refresh",
        value=st.session_state.get('auto_refresh', True),
        help="Otomatis refresh status sistem"
    )
    st.session_state.auto_refresh = auto_refresh
    
    # Show debug info
    show_debug = st.checkbox(
        "🐛 Debug Mode",
        value=st.session_state.get('show_debug', False),
        help="Tampilkan informasi debug dan metadata"
    )
    st.session_state.show_debug = show_debug
    
    # Max results slider
    max_results = st.slider(
        "📊 Maksimal Hasil",
        min_value=1,
        max_value=10,
        value=5,
        help="Jumlah maksimal dokumen sumber yang ditampilkan"
    )
    
    return search_type, use_enhanced_query, max_results

def render_graph_section():
    """Render graph-related features"""
    st.header("🕸️ Knowledge Graph")
    
    if st.session_state.graph_initialized:
        # Get graph stats
        graph_stats = APIClient.get_graph_stats()
        if graph_stats and graph_stats.get('has_data'):
            st.session_state.graph_stats = graph_stats
            
            # Display graph metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🔗 Nodes", graph_stats.get('nodes', 0))
            with col2:
                st.metric("🔄 Relations", graph_stats.get('relationships', 0))
            
            # Show node and relationship types
            node_types = graph_stats.get('node_types', [])
            if node_types:
                st.caption(f"Node types: {', '.join(node_types[:5])}")
            
            relationship_types = graph_stats.get('relationship_types', [])
            if relationship_types:
                st.caption(f"Relationship types: {', '.join(relationship_types[:5])}")
            
            # Graph visualization button
            if st.button("📊 Create Visualization", use_container_width=True):
                with st.spinner("Creating graph visualization..."):
                    result = APIClient.create_graph_visualization()
                    if result:
                        st.success("✅ Visualization created!")
                        st.info(f"File: {result.get('file_path', 'graph_visualization.html')}")
                    else:
                        st.error("❌ Failed to create visualization")
        else:
            st.info("📊 No graph data available")
    else:
        st.info("⚠️ Graph not initialized. Upload documents with graph processing enabled.")

def render_actions():
    """Render action buttons"""
    st.header("🎯 Aksi")
    
    # Clear history button
    if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
        result = APIClient.clear_chat_history()
        if result:
            st.success("✅ History berhasil dihapus")
            st.session_state.chat_history = []
            st.session_state.last_chunks = []
            st.session_state.last_response = {}
            st.rerun()
    
    # Refresh button
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()

def render_chat_interface(search_type: SearchType, use_enhanced_query: bool):
    """Render the main chat interface"""
    st.header("💬 Chat Assistant")
    
    # Show current search mode
    search_mode_text = {
        SearchType.VECTOR: "📊 Vector Search",
        SearchType.HYBRID: "🔗 Hybrid Search", 
        SearchType.GRAPH: "🕸️ Graph Search"
    }
    st.caption(f"Current mode: {search_mode_text.get(search_type, 'Unknown')}")
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Tanya tentang peraturan hukum:",
            placeholder="Contoh: Apa itu peraturan OJK tentang fintech?",
            key="chat_input"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submitted = st.form_submit_button("📤 Kirim", type="primary", use_container_width=True)
        with col2:
            if st.session_state.get('show_debug', False):
                debug_submitted = st.form_submit_button("🔍 Debug", use_container_width=True)
    
    # Process chat input
    if submitted and user_input:
        if not st.session_state.system_initialized:
            st.error("⚠️ Upload dokumen terlebih dahulu!")
        else:
            process_chat_message(user_input, search_type, use_enhanced_query)
    
    # Display chat history
    display_chat_history()

def process_chat_message(user_input: str, search_type: SearchType, use_enhanced_query: bool):
    """Process a chat message"""
    with st.spinner("🔍 Mencari jawaban..."):
        start_time = time.time()
        response = APIClient.send_chat_message(user_input, search_type, use_enhanced_query)
        
        if response and "answer" in response:
            processing_time = time.time() - start_time
            st.session_state.processing_time = processing_time
            st.session_state.last_response = response
            
            # Add messages to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now(),
                "search_type": search_type.value
            })
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"],
                "timestamp": datetime.now(),
                "processing_time": processing_time,
                "enhanced_query": use_enhanced_query,
                "generated_question": response.get("generated_question"),
                "enhanced_query_used": response.get("enhanced_query"),
                "search_type_used": response.get("search_type_used"),
                "tokens_used": response.get("tokens_used"),
                "graph_entities": response.get("graph_entities", []),
                "graph_relationships": response.get("graph_relationships", [])
            })
            
            # Store last chunks
            st.session_state.last_chunks = response.get("source_documents", [])
            
            # Show success with additional info
            success_msg = f"✅ Jawaban ditemukan dalam {processing_time:.2f}s"
            if response.get("search_type_used"):
                success_msg += f" ({response['search_type_used']} search)"
            if response.get("tokens_used"):
                success_msg += f" | {response['tokens_used']} tokens"
            
            st.success(success_msg)
            st.rerun()
        else:
            st.error("❌ Gagal mendapatkan jawaban")

def display_chat_history():
    """Display chat history with improved formatting"""
    st.markdown("---")
    
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            timestamp = message.get('timestamp', datetime.now())
            
            if message["role"] == "user":
                search_type = message.get('search_type', 'vector')
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 Anda</strong> <small>({timestamp.strftime('%H:%M:%S')}) <span class="metadata-item">{search_type}</span></small><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            
            else:
                processing_info = ""
                if message.get('processing_time'):
                    processing_info = f" ({message['processing_time']:.2f}s"
                    if message.get('search_type_used'):
                        processing_info += f", {message['search_type_used']}"
                    if message.get('enhanced_query'):
                        processing_info += ", Enhanced"
                    if message.get('tokens_used'):
                        processing_info += f", {message['tokens_used']} tokens"
                    processing_info += ")"
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant</strong> <small>({timestamp.strftime('%H:%M:%S')}{processing_info})</small><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show graph entities and relationships if available
                if message.get('graph_entities') or message.get('graph_relationships'):
                    with st.expander("🕸️ Graph Information"):
                        if message.get('graph_entities'):
                            st.markdown("**Entities Found:**")
                            entities_html = ""
                            for entity in message['graph_entities']:
                                entities_html += f'<span class="entity-badge">{entity}</span>'
                            st.markdown(entities_html, unsafe_allow_html=True)
                        
                        if message.get('graph_relationships'):
                            st.markdown("**Relationships Found:**")
                            relationships_html = ""
                            for rel in message['graph_relationships']:
                                relationships_html += f'<span class="relationship-badge">{rel}</span>'
                            st.markdown(relationships_html, unsafe_allow_html=True)
                
                # Show debug info if enabled
                if st.session_state.get('show_debug', False):
                    with st.expander("🐛 Debug Info"):
                        if message.get('generated_question'):
                            st.write(f"**Generated Question:** {message['generated_question']}")
                        if message.get('enhanced_query_used'):
                            st.write(f"**Enhanced Query Used:** {message['enhanced_query_used']}")
                        if message.get('search_type_used'):
                            st.write(f"**Search Type Used:** {message['search_type_used']}")
                        if message.get('tokens_used'):
                            st.write(f"**Tokens Used:** {message['tokens_used']}")
            
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("💬 Mulai percakapan dengan menanyakan sesuatu tentang dokumen hukum!")

def render_information_panel(max_results: int):
    """Render the information panel"""
    st.header("📊 Informasi")
    
    # System metrics
    if st.session_state.system_stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💬 Pesan", len(st.session_state.chat_history))
        with col2:
            st.metric("📄 Dokumen", st.session_state.system_stats.get('document_count', 0))
        with col3:
            st.metric("🔗 Chunks", st.session_state.system_stats.get('chunk_count', 0))
    
    # Graph metrics
    if st.session_state.graph_stats:
        st.subheader("🕸️ Graph Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nodes", st.session_state.graph_stats.get('nodes', 0))
        with col2:
            st.metric("Relations", st.session_state.graph_stats.get('relationships', 0))
    
    # Last response info
    if st.session_state.last_response:
        st.subheader("🔍 Last Query Info")
        response = st.session_state.last_response
        
        col1, col2 = st.columns(2)
        with col1:
            if response.get('processing_time'):
                st.metric("⏱️ Time", f"{response['processing_time']:.2f}s")
        with col2:
            if response.get('tokens_used'):
                st.metric("🔤 Tokens", response['tokens_used'])
        
        if response.get('search_type_used'):
            st.caption(f"Search type used: {response['search_type_used']}")
    
    # Last chunks button
    if st.button("📋 Lihat Chunks Terakhir", use_container_width=True):
        chunks = APIClient.get_last_chunks()
        if chunks:
            st.session_state.last_chunks = chunks
            st.success(f"✅ Loaded {len(chunks)} chunks")
    
    # Display chunks
    display_chunks(max_results)

def display_chunks(max_results: int):
    """Display retrieved chunks with improved formatting"""
    if st.session_state.last_chunks:
        st.subheader("📄 Chunks yang Digunakan:")
        
        # Limit display based on max_results
        chunks_to_show = st.session_state.last_chunks[:max_results]
        
        for i, chunk in enumerate(chunks_to_show):
            # Calculate display score
            rerank_score = chunk.get('rerank_score')
            regular_score = chunk.get('score')
            
            score_text = ""
            if rerank_score is not None:
                score_text = f"Rerank Score: {rerank_score:.3f}"
            elif regular_score is not None:
                score_text = f"Score: {regular_score:.3f}"
            
            with st.expander(f"Chunk {i+1} - {score_text}" if score_text else f"Chunk {i+1}"):
                # Score badges
                if rerank_score is not None or regular_score is not None:
                    col1, col2 = st.columns(2)
                    if rerank_score is not None:
                        with col1:
                            st.markdown(f'<span class="score-badge">Rerank: {rerank_score:.3f}</span>', unsafe_allow_html=True)
                    if regular_score is not None:
                        with col2:
                            st.markdown(f'<span class="score-badge">Original: {regular_score:.3f}</span>', unsafe_allow_html=True)
                
                # Content preview
                content = chunk.get("content", "")
                if len(content) > 500:
                    st.text_area(
                        "Content Preview:",
                        content[:500] + "...",
                        height=100,
                        key=f"chunk_content_{i}",
                        disabled=True
                    )
                    
                    # Show full content button
                    if st.button(f"📖 Lihat Selengkapnya", key=f"show_full_{i}"):
                        st.text_area(
                            "Full Content:",
                            content,
                            height=200,
                            key=f"full_content_{i}",
                            disabled=True
                        )
                else:
                    st.text_area(
                        "Content:",
                        content,
                        height=100,
                        key=f"chunk_content_short_{i}",
                        disabled=True
                    )
                
                # Metadata
                metadata = chunk.get("metadata", {})
                if metadata:
                    st.markdown("**Metadata:**")
                    
                    # Format metadata nicely
                    metadata_cols = st.columns(2)
                    col_idx = 0
                    
                    for key, value in metadata.items():
                        if key not in ['score', 'rerank_score', 'content_hash']:  # Skip scores as they're shown above
                            with metadata_cols[col_idx % 2]:
                                if key == 'source':
                                    st.markdown(f'<span class="metadata-item">📄 {key}: {value}</span>', unsafe_allow_html=True)
                                elif key == 'file_type':
                                    st.markdown(f'<span class="metadata-item">📋 {key}: {value}</span>', unsafe_allow_html=True)
                                elif key == 'chunk_id':
                                    st.markdown(f'<span class="metadata-item">🔗 {key}: {value}</span>', unsafe_allow_html=True)
                                elif key == 'chunk_length':
                                    st.markdown(f'<span class="metadata-item">📏 {key}: {value}</span>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<span class="metadata-item">{key}: {value}</span>', unsafe_allow_html=True)
                            col_idx += 1
        
        # Show pagination info if there are more chunks
        if len(st.session_state.last_chunks) > max_results:
            st.info(f"Menampilkan {max_results} dari {len(st.session_state.last_chunks)} chunks. Ubah 'Maksimal Hasil' di pengaturan untuk melihat lebih banyak.")
    
    else:
        st.info("💡 Tidak ada chunks yang ditampilkan. Ajukan pertanyaan untuk melihat dokumen yang digunakan.")

def render_graph_visualization():
    """Render graph visualization section"""
    if st.session_state.graph_initialized:
        st.header("🕸️ Graph Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Create New Visualization", use_container_width=True):
                with st.spinner("Creating graph visualization..."):
                    result = APIClient.create_graph_visualization()
                    if result:
                        st.success("✅ Visualization created!")
                        st.info(f"File: {result.get('file_path', 'graph_visualization.html')}")
                        
                        # You could add logic here to display the visualization
                        # For now, we'll just show the file path
                    else:
                        st.error("❌ Failed to create visualization")
        
        with col2:
            st.caption("Graph visualization will be saved as HTML file that you can open in your browser.")
        
        # Show current graph stats
        if st.session_state.graph_stats:
            stats = st.session_state.graph_stats
            
            st.subheader("📈 Current Graph Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🔗 Total Nodes", stats.get('nodes', 0))
                if stats.get('node_types'):
                    st.write("**Node Types:**")
                    for node_type in stats.get('node_types', [])[:10]:  # Show max 10
                        st.write(f"• {node_type}")
            
            with col2:
                st.metric("🔄 Total Relationships", stats.get('relationships', 0))
                if stats.get('relationship_types'):
                    st.write("**Relationship Types:**")
                    for rel_type in stats.get('relationship_types', [])[:10]:  # Show max 10
                        st.write(f"• {rel_type}")

def display_advanced_search_info():
    """Display information about different search types"""
    st.subheader("🔍 Search Types Information")
    
    with st.expander("📊 Vector Search"):
        st.write("""
        **Vector Search** menggunakan embedding untuk mencari dokumen berdasarkan kesamaan semantik.
        - ✅ Selalu tersedia
        - ✅ Cepat dan efisien
        - ✅ Baik untuk pencarian umum
        - ✅ Tidak memerlukan graph processing
        """)
    
    if st.session_state.graph_initialized:
        with st.expander("🔗 Hybrid Search"):
            st.write("""
            **Hybrid Search** menggabungkan vector search dengan knowledge graph.
            - ✅ Menggabungkan kecepatan vector dengan konteks graph
            - ✅ Lebih akurat untuk pertanyaan kompleks
            - ✅ Memanfaatkan hubungan antar entitas
            - ⚠️ Memerlukan graph data
            """)
        
        with st.expander("🕸️ Graph Search"):
            st.write("""
            **Graph Search** fokus pada hubungan semantik dalam knowledge graph.
            - ✅ Excellent untuk pertanyaan relasional
            - ✅ Menemukan koneksi tersembunyi
            - ✅ Memberikan konteks yang kaya
            - ⚠️ Memerlukan graph data yang berkualitas
            """)
    else:
        with st.expander("🔒 Advanced Search (Locked)"):
            st.write("""
            **Hybrid & Graph Search** tidak tersedia karena:
            - 📄 Belum ada dokumen yang diupload dengan Graph Processing
            - 🔄 Upload dokumen dan aktifkan "Enable Graph Processing"
            - 🕸️ Sistem akan memproses dokumen menjadi knowledge graph
            
            **Untuk mengaktifkan:**
            1. Upload dokumen baru dengan ✅ Graph Processing
            2. Atau reprocess dokumen existing (hapus data dan upload ulang)
            """)

def render_performance_metrics():
    """Display performance metrics and statistics"""
    if st.session_state.system_stats or st.session_state.last_response:
        st.subheader("📈 Performance Metrics")
        
        # Current session metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💬 Messages",
                len(st.session_state.chat_history),
                help="Total messages in current session"
            )
        
        with col2:
            if st.session_state.last_response and st.session_state.last_response.get('processing_time'):
                st.metric(
                    "⏱️ Last Query",
                    f"{st.session_state.last_response['processing_time']:.2f}s",
                    help="Processing time for last query"
                )
        
        with col3:
            if st.session_state.last_response and st.session_state.last_response.get('tokens_used'):
                st.metric(
                    "🔤 Tokens Used",
                    st.session_state.last_response['tokens_used'],
                    help="Tokens used in last query"
                )
        
        with col4:
            if st.session_state.last_response and st.session_state.last_response.get('search_type_used'):
                st.metric(
                    "🔍 Search Type",
                    st.session_state.last_response['search_type_used'].upper(),
                    help="Search type used in last query"
                )
        
        # System statistics
        if st.session_state.system_stats:
            stats = st.session_state.system_stats
            
            with st.expander("📊 Detailed System Stats"):
                st.json({
                    "system_initialized": stats.get('system_initialized'),
                    "graph_initialized": stats.get('graph_initialized'),
                    "document_count": stats.get('document_count'),
                    "chunk_count": stats.get('chunk_count'),
                    "last_upload_time": str(stats.get('last_upload_time')),
                    "search_types_available": stats.get('search_types_available'),
                    "graph_stats": stats.get('graph_stats', {})
                })

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Title with enhanced features indicator
    title_text = "⚖️ Legal RAG Assistant"
    if st.session_state.graph_initialized:
        title_text += " 🕸️"
    
    st.title(title_text)
    
    # Subtitle with current status
    if st.session_state.system_initialized:
        if st.session_state.graph_initialized:
            st.caption("✅ System ready with Graph support | Choose your search method below")
        else:
            st.caption("✅ System ready with Vector search | Upload documents with graph processing for advanced features")
    else:
        st.caption("⚠️ Please upload documents to get started")
    
    st.markdown("---")
    
    # Render sidebar and get settings
    search_type, use_enhanced_query, max_results = render_sidebar()
    
    # Main layout
    main_col, info_col = st.columns([2, 1])
    
    with main_col:
        render_chat_interface(search_type, use_enhanced_query)
    
    with info_col:
        # Tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📊 Stats", "📄 Chunks", "🔍 Search"])
        
        with tab1:
            render_performance_metrics()
            
            # Graph visualization section
            if st.session_state.graph_initialized:
                st.markdown("---")
                render_graph_visualization()
        
        with tab2:
            # Last chunks button
            if st.button("📋 Refresh Chunks", use_container_width=True):
                chunks = APIClient.get_last_chunks()
                if chunks:
                    st.session_state.last_chunks = chunks
                    st.success(f"✅ Loaded {len(chunks)} chunks")
            
            # Display chunks
            display_chunks(max_results)
        
        with tab3:
            display_advanced_search_info()
    
    # Footer with additional info
    st.markdown("---")
    
    # Show system status in footer
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        if st.session_state.system_initialized:
            st.success("🟢 System Online")
        else:
            st.error("🔴 System Offline")
    
    with footer_col2:
        if st.session_state.graph_initialized:
            st.info("🕸️ Graph Enabled")
        else:
            st.warning("📊 Vector Only")
    
    with footer_col3:
        # Show current search type
        current_search = st.session_state.get('search_type', SearchType.VECTOR)
        search_emoji = {
            SearchType.VECTOR: "📊",
            SearchType.HYBRID: "🔗", 
            SearchType.GRAPH: "🕸️"
        }
        st.info(f"{search_emoji.get(current_search, '🔍')} {current_search.value.title()} Mode")
    
    # Auto-refresh functionality
    if st.session_state.get('auto_refresh', True):
        # Auto refresh every 30 seconds (optional)
        time.sleep(0.1)

if __name__ == "__main__":
    main()