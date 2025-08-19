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
    page_title="Enhanced Legal RAG Assistant with Feedback Learning",
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
.enhanced-message {
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
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
.feedback-badge {
    background-color: #e1f5fe;
    color: #0277bd;
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
.feedback-section {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid #28a745;
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
    if 'feedback_stats' not in st.session_state:
        st.session_state.feedback_stats = {}
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False
    if 'search_type' not in st.session_state:
        st.session_state.search_type = SearchType.VECTOR
    if 'last_response' not in st.session_state:
        st.session_state.last_response = {}
    if 'show_feedback' not in st.session_state:
        st.session_state.show_feedback = True
    if 'pending_feedback' not in st.session_state:
        st.session_state.pending_feedback = None

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
        """Send chat message to the enhanced API endpoint"""
        payload = {
            "question": question,
            "search_type": search_type.value,
            "use_enhanced_query": use_enhanced_query
        }
        # Use the enhanced chat endpoint
        return APIClient._make_request("POST", "/chat", json=payload)
    
    @staticmethod
    def send_feedback(query: str, response: str, relevance_score: int, quality_score: int, 
                     response_time: Optional[float] = None, search_type: Optional[str] = None,
                     comments: Optional[str] = None, user_id: Optional[str] = None) -> Optional[Dict]:
        """Send feedback to the API"""
        payload = {
            "query": query,
            "response": response,
            "relevance_score": relevance_score,
            "quality_score": quality_score,
            "response_time": response_time,
            "search_type": search_type,
            "comments": comments,
            "user_id": user_id
        }
        return APIClient._make_request("POST", "/feedback", json=payload)
    
    @staticmethod
    def get_feedback_stats() -> Optional[Dict]:
        """Get feedback statistics"""
        return APIClient._make_request("GET", "/feedback/stats")
    
    @staticmethod
    def clear_feedback_history() -> Optional[Dict]:
        """Clear feedback history"""
        return APIClient._make_request("DELETE", "/feedback/history")
    
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
            - ✅ **Enabled**: Mendukung Hybrid & Graph search + Feedback Learning
            - ❌ **Disabled**: Hanya Vector search, lebih cepat
            
            **File Types:**
            - PDF documents
            - Text files (.txt)
            - ZIP archives (akan diekstrak)
            
            **Best Practices:**
            - Upload legal documents untuk hasil terbaik
            - File berukuran wajar (< 10MB per file)
            - Gunakan Graph Processing untuk analisis relasi
            - Berikan feedback untuk meningkatkan akurasi sistem
            """)
        
        st.markdown("---")
        
        # System Status
        render_system_status()
        
        st.markdown("---")
        
        # Settings
        search_type, use_enhanced_query, max_results = render_settings()
        
        st.markdown("---")
        
        # Feedback Section
        render_feedback_section()
        
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

def render_feedback_section():
    """Render feedback learning section"""
    st.header("🎯 Feedback Learning")
    
    # Get feedback stats
    feedback_stats = APIClient.get_feedback_stats()
    if feedback_stats:
        st.session_state.feedback_stats = feedback_stats
        
        # Show feedback metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Total Feedback", feedback_stats.get('total_feedback', 0))
        with col2:
            avg_score = (feedback_stats.get('average_relevance', 0) + feedback_stats.get('average_quality', 0)) / 2
            st.metric("⭐ Avg Score", f"{avg_score:.1f}/5")
        
        # Show recent feedback trends
        if feedback_stats.get('total_feedback', 0) > 0:
            st.success("🧠 Feedback Learning Active")
            
            # Search type distribution
            search_dist = feedback_stats.get('search_type_distribution', {})
            if search_dist:
                st.caption("Search type usage: " + ", ".join([f"{k}: {v}" for k, v in search_dist.items()]))
        else:
            st.info("💡 Berikan feedback untuk meningkatkan akurasi")
    
    # Feedback toggle
    show_feedback = st.checkbox(
        "📝 Show Feedback Forms",
        value=st.session_state.get('show_feedback', True),
        help="Tampilkan form feedback setelah setiap respons"
    )
    st.session_state.show_feedback = show_feedback
    
    # Clear feedback button
    if st.button("🗑️ Clear Feedback History", help="Hapus semua feedback (use with caution)"):
        result = APIClient.clear_feedback_history()
        if result:
            st.success("✅ Feedback history cleared")
            st.session_state.feedback_stats = {}
            st.rerun()

def render_settings():
    """Render settings panel"""
    st.header("⚙️ Pengaturan")
    
    # Search type selection - only show available options
    search_options = ["Vector Search"]
    search_values = [SearchType.VECTOR]
    
    if st.session_state.graph_initialized:
        search_options.extend(["Hybrid Search (with Feedback)", "Graph Search"])
        search_values.extend([SearchType.HYBRID, SearchType.GRAPH])
    else:
        st.info("🕸️ Upload documents with Graph Processing enabled to unlock advanced search options and feedback learning")
    
    search_type_idx = st.selectbox(
        "🔍 Tipe Pencarian",
        range(len(search_options)),
        format_func=lambda x: search_options[x],
        index=0,
        help="Pilih metode pencarian berdasarkan ketersediaan data"
    )
    
    search_type = search_values[search_type_idx]
    st.session_state.search_type = search_type
    
    # Show search type explanations with feedback info
    if search_type == SearchType.VECTOR:
        st.caption("📊 Pencarian similarity berbasis embedding vector")
        if not st.session_state.graph_initialized:
            st.caption("💡 Graph search tidak tersedia - upload dengan Graph Processing untuk fitur feedback learning")
    elif search_type == SearchType.HYBRID:
        st.caption("🔗 Kombinasi vector search + knowledge graph + feedback learning")
        if st.session_state.feedback_stats.get('total_feedback', 0) > 0:
            st.markdown(f'<span class="feedback-badge">🧠 Learning dari {st.session_state.feedback_stats["total_feedback"]} feedback</span>', unsafe_allow_html=True)
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
            st.session_state.pending_feedback = None
            st.rerun()
    
    # Refresh button
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()

def render_feedback_form(message_data: Dict, message_index: int):
    """Render feedback form for a specific message"""
    if not st.session_state.get('show_feedback', True):
        return
    
    # Create unique key for this feedback form
    form_key = f"feedback_form_{message_index}"
    
    with st.expander("📝 Rate this response", expanded=False):
        with st.form(form_key):
            st.write("Help improve the system by rating this response:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                relevance_score = st.slider(
                    "🎯 Relevance (How relevant was this answer?)",
                    min_value=1, max_value=5, value=3,
                    help="1 = Not relevant, 5 = Very relevant",
                    key=f"relevance_{message_index}"
                )
            
            with col2:
                quality_score = st.slider(
                    "⭐ Quality (How good was this answer?)",
                    min_value=1, max_value=5, value=3,
                    help="1 = Poor quality, 5 = Excellent quality",
                    key=f"quality_{message_index}"
                )
            
            comments = st.text_area(
                "💬 Additional comments (optional)",
                placeholder="What could be improved? Was the answer accurate?",
                key=f"comments_{message_index}"
            )
            
            if st.form_submit_button("📤 Submit Feedback"):
                # Find the corresponding user question
                user_message = None
                if message_index > 0:
                    user_message = st.session_state.chat_history[message_index - 1]
                
                if user_message and user_message.get('role') == 'user':
                    # Submit feedback
                    result = APIClient.send_feedback(
                        query=user_message['content'],
                        response=message_data['content'],
                        relevance_score=relevance_score,
                        quality_score=quality_score,
                        response_time=message_data.get('processing_time'),
                        search_type=message_data.get('search_type_used'),
                        comments=comments if comments else None,
                        user_id="streamlit_user"  # You could make this configurable
                    )
                    
                    if result and result.get('success'):
                        st.success("✅ Thank you for your feedback! This will help improve future responses.")
                        st.rerun()
                    else:
                        st.error("❌ Failed to submit feedback. Please try again.")
                else:
                    st.error("❌ Could not find the corresponding question for this feedback.")

def render_chat_interface(search_type: SearchType, use_enhanced_query: bool):
    """Render the main chat interface"""
    st.header("💬 Enhanced Chat Assistant")
    
    # Show current search mode with feedback info
    search_mode_text = {
        SearchType.VECTOR: "📊 Vector Search",
        SearchType.HYBRID: "🔗 Hybrid Search + Feedback Learning", 
        SearchType.GRAPH: "🕸️ Graph Search"
    }
    st.caption(f"Current mode: {search_mode_text.get(search_type, 'Unknown')}")
    
    # Show feedback learning status
    if search_type in [SearchType.HYBRID, SearchType.VECTOR] and st.session_state.feedback_stats.get('total_feedback', 0) > 0:
        feedback_count = st.session_state.feedback_stats['total_feedback']
        st.markdown(f'<span class="feedback-badge">🧠 Learning from {feedback_count} feedback entries</span>', unsafe_allow_html=True)
    
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
                "graph_relationships": response.get("graph_relationships", []),
                # Enhanced response fields
                "feedback_learning_applied": response.get("feedback_learning_applied", False),
                "feedback_entries_used": response.get("feedback_entries_used", 0),
                "documents_learned": response.get("documents_learned", 0),
                "query_with_feedback_time": response.get("query_with_feedback_time")
            })
            
            # Store last chunks
            st.session_state.last_chunks = response.get("source_documents", [])
            
            # Show success with additional info including feedback learning
            success_msg = f"✅ Jawaban ditemukan dalam {processing_time:.2f}s"
            if response.get("search_type_used"):
                success_msg += f" ({response['search_type_used']} search)"
            if response.get("tokens_used"):
                success_msg += f" | {response['tokens_used']} tokens"
            if response.get("feedback_learning_applied"):
                success_msg += f" | 🧠 Learned from {response.get('feedback_entries_used', 0)} feedback"
            
            st.success(success_msg)
            st.rerun()
        else:
            st.error("❌ Gagal mendapatkan jawaban")

def display_chat_history():
    """Display chat history with improved formatting and feedback forms"""
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
                
                # Add feedback learning indicator
                feedback_indicator = ""
                if message.get('feedback_learning_applied'):
                    feedback_entries = message.get('feedback_entries_used', 0)
                    documents_learned = message.get('documents_learned', 0)
                    feedback_indicator = f'<span class="feedback-badge">🧠 Learned: {feedback_entries} feedback, {documents_learned} docs</span><br>'
                
                message_class = "assistant-message"
                if message.get('feedback_learning_applied'):
                    message_class = "enhanced-message"
                
                st.markdown(f"""
                <div class="chat-message {message_class}">
                    <strong>🤖 Assistant</strong> <small>({timestamp.strftime('%H:%M:%S')}{processing_info})</small><br>
                    {feedback_indicator}
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show enhanced feedback learning info if available
                if message.get('feedback_learning_applied') and st.session_state.get('show_debug', False):
                    with st.expander("🧠 Feedback Learning Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📝 Feedback Used", message.get('feedback_entries_used', 0))
                        with col2:
                            st.metric("📄 Docs Learned", message.get('documents_learned', 0))
                        with col3:
                            if message.get('query_with_feedback_time'):
                                st.metric("⏱️ Learning Time", f"{message['query_with_feedback_time']:.3f}s")
                
                # Show graph entities and relationships if available
                if message.get('graph_entities') or message.get('graph_relationships'):
                    with st.expander("🕸️ Graph Information"):
                        if message.get('graph_entities'):
                            st.markdown("**Entities Found:**")
                            entities_html = ""
                            for entity in message['graph_entities']:
                                entity_text = entity.get('id', str(entity)) if isinstance(entity, dict) else str(entity)
                                entities_html += f'<span class="entity-badge">{entity_text}</span>'
                            st.markdown(entities_html, unsafe_allow_html=True)
                        
                        if message.get('graph_relationships'):
                            st.markdown("**Relationships Found:**")
                            relationships_html = ""
                            for rel in message['graph_relationships']:
                                rel_text = f"{rel.get('source', '')}-{rel.get('type', '')}-{rel.get('target', '')}" if isinstance(rel, dict) else str(rel)
                                relationships_html += f'<span class="relationship-badge">{rel_text}</span>'
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
                        if message.get('feedback_learning_applied'):
                            st.write(f"**Feedback Learning Applied:** {message['feedback_learning_applied']}")
                            st.write(f"**Feedback Entries Used:** {message.get('feedback_entries_used', 0)}")
                            st.write(f"**Documents Learned:** {message.get('documents_learned', 0)}")
                
                # Render feedback form for assistant messages
                render_feedback_form(message, i)
            
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("💬 Mulai percakapan dengan menanyakan sesuatu tentang dokumen hukum!")
        st.info("🧠 Sistem akan belajar dari feedback Anda untuk memberikan jawaban yang lebih baik!")

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
    
    # Feedback metrics
    if st.session_state.feedback_stats:
        st.subheader("🧠 Feedback Learning")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Total Feedback", st.session_state.feedback_stats.get('total_feedback', 0))
        with col2:
            st.metric("🎯 Avg Relevance", f"{st.session_state.feedback_stats.get('average_relevance', 0):.1f}/5")
        with col3:
            st.metric("⭐ Avg Quality", f"{st.session_state.feedback_stats.get('average_quality', 0):.1f}/5")
        
        # Show search type distribution
        search_dist = st.session_state.feedback_stats.get('search_type_distribution', {})
        if search_dist:
            st.caption("Feedback by search type:")
            for search_type, count in search_dist.items():
                st.caption(f"• {search_type}: {count} feedback")
    
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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if response.get('processing_time'):
                st.metric("⏱️ Time", f"{response['processing_time']:.2f}s")
        with col2:
            if response.get('tokens_used'):
                st.metric("🔤 Tokens", response['tokens_used'])
        with col3:
            if response.get('feedback_learning_applied'):
                st.metric("🧠 Feedback Used", response.get('feedback_entries_used', 0))
        
        if response.get('search_type_used'):
            st.caption(f"Search type used: {response['search_type_used']}")
        
        # Show feedback learning details
        if response.get('feedback_learning_applied'):
            st.success("🧠 Feedback learning was applied to this query!")
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"📝 Feedback entries used: {response.get('feedback_entries_used', 0)}")
            with col2:
                st.caption(f"📄 Documents learned: {response.get('documents_learned', 0)}")
    
    # Last chunks button
    if st.button("📋 Lihat Chunks Terakhir", use_container_width=True):
        chunks = APIClient.get_last_chunks()
        if chunks:
            st.session_state.last_chunks = chunks
            st.success(f"✅ Loaded {len(chunks)} chunks")
    
    # Display chunks
    display_chunks(max_results)

def display_chunks(max_results: int):
    """Display retrieved chunks with improved formatting including feedback scores"""
    if st.session_state.last_chunks:
        st.subheader("📄 Chunks yang Digunakan:")
        
        # Limit display based on max_results
        chunks_to_show = st.session_state.last_chunks[:max_results]
        
        for i, chunk in enumerate(chunks_to_show):
            # Calculate display score
            rerank_score = chunk.get('rerank_score')
            regular_score = chunk.get('score')
            relevance_score = chunk.get('metadata', {}).get('relevance_score', 1.0) if isinstance(chunk, dict) else getattr(chunk, 'metadata', {}).get('relevance_score', 1.0)
            feedback_applied = chunk.get('metadata', {}).get('feedback_applied', 0) if isinstance(chunk, dict) else getattr(chunk, 'metadata', {}).get('feedback_applied', 0)
            
            score_text = ""
            if rerank_score is not None:
                score_text = f"Rerank: {rerank_score:.3f}"
            elif regular_score is not None:
                score_text = f"Score: {regular_score:.3f}"
            
            # Add feedback info to title
            feedback_text = ""
            if feedback_applied > 0:
                feedback_text = f" (🧠 {feedback_applied} feedback applied)"
            
            with st.expander(f"Chunk {i+1} - {score_text}{feedback_text}" if score_text else f"Chunk {i+1}{feedback_text}"):
                # Score badges including feedback learning
                score_cols = st.columns(4)
                
                with score_cols[0]:
                    if rerank_score is not None:
                        st.markdown(f'<span class="score-badge">Rerank: {rerank_score:.3f}</span>', unsafe_allow_html=True)
                
                with score_cols[1]:
                    if regular_score is not None:
                        st.markdown(f'<span class="score-badge">Original: {regular_score:.3f}</span>', unsafe_allow_html=True)
                
                with score_cols[2]:
                    if relevance_score != 1.0:  # Only show if different from default
                        st.markdown(f'<span class="feedback-badge">Relevance: {relevance_score:.3f}</span>', unsafe_allow_html=True)
                
                with score_cols[3]:
                    if feedback_applied > 0:
                        st.markdown(f'<span class="feedback-badge">🧠 {feedback_applied} feedback</span>', unsafe_allow_html=True)
                
                # Content preview
                content = chunk.get("content", "") if isinstance(chunk, dict) else getattr(chunk, 'page_content', "")
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
                metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else getattr(chunk, 'metadata', {})
                if metadata:
                    st.markdown("**Metadata:**")
                    
                    # Format metadata nicely
                    metadata_cols = st.columns(2)
                    col_idx = 0
                    
                    for key, value in metadata.items():
                        if key not in ['score', 'rerank_score', 'content_hash', 'relevance_score', 'feedback_applied']:  # Skip scores as they're shown above
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
    """Display information about different search types including feedback learning"""
    st.subheader("🔍 Search Types Information")
    
    with st.expander("📊 Vector Search"):
        st.write("""
        **Vector Search** menggunakan embedding untuk mencari dokumen berdasarkan kesamaan semantik.
        - ✅ Selalu tersedia
        - ✅ Cepat dan efisien
        - ✅ Baik untuk pencarian umum
        - ✅ Tidak memerlukan graph processing
        - 🧠 Basic feedback learning (tanpa graph context)
        """)
    
    if st.session_state.graph_initialized:
        with st.expander("🔗 Hybrid Search (Recommended)"):
            feedback_count = st.session_state.feedback_stats.get('total_feedback', 0)
            st.write(f"""
            **Hybrid Search** menggabungkan vector search dengan knowledge graph dan feedback learning.
            - ✅ Menggabungkan kecepatan vector dengan konteks graph
            - ✅ Lebih akurat untuk pertanyaan kompleks
            - ✅ Memanfaatkan hubungan antar entitas
            - 🧠 **Advanced feedback learning** - sistem belajar dari {feedback_count} feedback
            - 🎯 Document relevance adjustment berdasarkan feedback history
            - 📈 Semakin banyak feedback, semakin akurat jawabannya
            - ⚠️ Memerlukan graph data
            """)
        
        with st.expander("🕸️ Graph Search"):
            st.write("""
            **Graph Search** fokus pada hubungan semantik dalam knowledge graph.
            - ✅ Excellent untuk pertanyaan relasional
            - ✅ Menemukan koneksi tersembunyi
            - ✅ Memberikan konteks yang kaya
            - 📊 Tidak menggunakan feedback learning
            - ⚠️ Memerlukan graph data yang berkualitas
            """)
    else:
        with st.expander("🔒 Advanced Search & Feedback Learning (Locked)"):
            st.write("""
            **Hybrid & Graph Search dengan Feedback Learning** tidak tersedia karena:
            - 📄 Belum ada dokumen yang diupload dengan Graph Processing
            - 🔄 Upload dokumen dan aktifkan "Enable Graph Processing"
            - 🕸️ Sistem akan memproses dokumen menjadi knowledge graph
            - 🧠 Feedback learning memerlukan graph context untuk optimal performance
            
            **Untuk mengaktifkan:**
            1. Upload dokumen baru dengan ✅ Graph Processing
            2. Sistem akan membangun knowledge graph
            3. Feedback learning akan aktif untuk Hybrid search
            4. Berikan rating pada jawaban untuk meningkatkan akurasi
            """)

def render_performance_metrics():
    """Display performance metrics and statistics including feedback learning metrics"""
    if st.session_state.system_stats or st.session_state.last_response or st.session_state.feedback_stats:
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
                search_type = st.session_state.last_response['search_type_used'].upper()
                if st.session_state.last_response.get('feedback_learning_applied'):
                    search_type += " 🧠"
                st.metric(
                    "🔍 Search Type",
                    search_type,
                    help="Search type used in last query"
                )
        
        # Feedback learning metrics
        if st.session_state.feedback_stats and st.session_state.feedback_stats.get('total_feedback', 0) > 0:
            st.markdown("**🧠 Feedback Learning Metrics:**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📝 Total Feedback", st.session_state.feedback_stats['total_feedback'])
            with col2:
                st.metric("🎯 Avg Relevance", f"{st.session_state.feedback_stats.get('average_relevance', 0):.1f}/5")
            with col3:
                st.metric("⭐ Avg Quality", f"{st.session_state.feedback_stats.get('average_quality', 0):.1f}/5")
            with col4:
                if st.session_state.last_response and st.session_state.last_response.get('feedback_learning_applied'):
                    st.metric("🧠 Last Learning", f"{st.session_state.last_response.get('feedback_entries_used', 0)} entries")
        
        # System statistics
        if st.session_state.system_stats:
            stats = st.session_state.system_stats
            
            with st.expander("📊 Detailed System Stats"):
                system_stats_display = {
                    "system_initialized": stats.get('system_initialized'),
                    "graph_initialized": stats.get('graph_initialized'),
                    "document_count": stats.get('document_count'),
                    "chunk_count": stats.get('chunk_count'),
                    "last_upload_time": str(stats.get('last_upload_time')),
                    "search_types_available": stats.get('search_types_available'),
                    "graph_stats": stats.get('graph_stats', {}),
                }
                
                # Add feedback stats if available
                if st.session_state.feedback_stats:
                    system_stats_display["feedback_stats"] = st.session_state.feedback_stats
                
                st.json(system_stats_display)

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Title with enhanced features indicator
    title_text = "⚖️ Enhanced Legal RAG Assistant"
    if st.session_state.graph_initialized:
        title_text += " 🕸️🧠"
    elif st.session_state.feedback_stats.get('total_feedback', 0) > 0:
        title_text += " 🧠"
    
    st.title(title_text)
    
    # Subtitle with current status
    if st.session_state.system_initialized:
        if st.session_state.graph_initialized:
            feedback_count = st.session_state.feedback_stats.get('total_feedback', 0)
            if feedback_count > 0:
                st.caption(f"✅ System ready with Graph support & Feedback Learning ({feedback_count} feedback entries) | Advanced AI that learns from your ratings")
            else:
                st.caption("✅ System ready with Graph support | Rate responses to enable feedback learning")
        else:
            st.caption("✅ System ready with Vector search | Upload documents with graph processing for advanced learning features")
    else:
        st.caption("⚠️ Please upload documents to get started with AI-powered legal assistance")
    
    st.markdown("---")
    
    # Render sidebar and get settings
    search_type, use_enhanced_query, max_results = render_sidebar()
    
    # Main layout
    main_col, info_col = st.columns([2, 1])
    
    with main_col:
        render_chat_interface(search_type, use_enhanced_query)
    
    with info_col:
        # Tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Stats", "📄 Chunks", "🔍 Search", "🧠 Learning"])
        
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
        
        with tab4:
            # Feedback Learning Tab
            st.subheader("🧠 Feedback Learning System")
            
            feedback_stats = st.session_state.feedback_stats
            if feedback_stats and feedback_stats.get('total_feedback', 0) > 0:
                st.success("✅ Feedback learning is active!")
                
                # Feedback overview
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📝 Total Feedback", feedback_stats['total_feedback'])
                    st.metric("🎯 Avg Relevance", f"{feedback_stats.get('average_relevance', 0):.1f}/5")
                with col2:
                    st.metric("⭐ Avg Quality", f"{feedback_stats.get('average_quality', 0):.1f}/5")
                    
                    # Calculate overall satisfaction
                    avg_satisfaction = (feedback_stats.get('average_relevance', 0) + feedback_stats.get('average_quality', 0)) / 2
                    satisfaction_color = "🟢" if avg_satisfaction >= 4 else "🟡" if avg_satisfaction >= 3 else "🔴"
                    st.metric("📈 Overall", f"{satisfaction_color} {avg_satisfaction:.1f}/5")
                
                # Search type feedback distribution
                search_dist = feedback_stats.get('search_type_distribution', {})
                if search_dist:
                    st.subheader("📊 Feedback by Search Type")
                    for search_type_name, count in search_dist.items():
                        st.write(f"• **{search_type_name.title()}**: {count} feedback entries")
                
                # Recent feedback trends (if available)
                recent_feedback = feedback_stats.get('recent_feedback', [])
                if recent_feedback:
                    st.subheader("📈 Recent Feedback Trends")
                    st.write("Latest feedback entries help the system learn and improve:")
                    for i, fb in enumerate(recent_feedback[-3:], 1):  # Show last 3
                        if isinstance(fb, dict):
                            query_preview = fb.get('query', '')[:50] + "..." if len(fb.get('query', '')) > 50 else fb.get('query', '')
                            relevance = fb.get('relevance_score', 0)
                            quality = fb.get('quality_score', 0)
                            st.write(f"{i}. **Query**: {query_preview}")
                            st.write(f"   📊 Scores: Relevance {relevance}/5, Quality {quality}/5")
                
                # How feedback learning works
                with st.expander("🎓 How Feedback Learning Works"):
                    st.write("""
                    **The system learns from your feedback to improve future responses:**
                    
                    1. **Document Relevance Adjustment**: Feedback scores adjust how relevant documents are ranked
                    2. **Query Similarity Matching**: Past feedback on similar queries influences current results
                    3. **Continuous Learning**: Each feedback entry helps the system understand your preferences
                    4. **Contextual Application**: Feedback is applied based on query and document similarity
                    
                    **Best Practices for Giving Feedback:**
                    - Rate both relevance (how well it answered your question) and quality (how well-written/accurate)
                    - Add comments for specific issues or praise
                    - Consistent rating helps the system learn your preferences
                    - Rate various types of questions to improve overall performance
                    """)
            else:
                st.info("🎯 No feedback data yet. Start giving ratings to responses to enable learning!")
                
                st.subheader("🚀 Getting Started with Feedback Learning")
                st.write("""
                **Steps to enable feedback learning:**
                
                1. **Ask Questions**: Start asking questions about your documents
                2. **Rate Responses**: Use the feedback forms after each assistant response
                3. **Be Consistent**: Rate both relevance and quality on a 1-5 scale
                4. **Add Comments**: Provide specific feedback about what was good or could be improved
                5. **See Improvements**: The system will learn from your feedback and provide better answers
                
                **Benefits:**
                - 🎯 More relevant document selection
                - 📈 Improved answer quality over time
                - 🧠 Personalized AI that understands your domain
                - 📊 Better matching for similar future questions
                """)
                
                # Sample feedback form (disabled)
                with st.expander("📝 Sample Feedback Form"):
                    st.write("This is what you'll see after each assistant response:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.slider("🎯 Relevance", 1, 5, 3, disabled=True)
                    with col2:
                        st.slider("⭐ Quality", 1, 5, 3, disabled=True)
                    st.text_area("💬 Comments", placeholder="What could be improved?", disabled=True)
                    st.button("📤 Submit Feedback", disabled=True)
    
    # Footer with additional info
    st.markdown("---")
    
    # Show system status in footer
    footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)
    
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
        feedback_count = st.session_state.feedback_stats.get('total_feedback', 0)
        if feedback_count > 0:
            st.info(f"🧠 Learning ({feedback_count})")
        else:
            st.warning("🎯 No Learning Data")
    
    with footer_col4:
        # Show current search type
        current_search = st.session_state.get('search_type', SearchType.VECTOR)
        search_emoji = {
            SearchType.VECTOR: "📊",
            SearchType.HYBRID: "🔗", 
            SearchType.GRAPH: "🕸️"
        }
        mode_text = f"{search_emoji.get(current_search, '🔍')} {current_search.value.title()}"
        if current_search == SearchType.HYBRID and feedback_count > 0:
            mode_text += " 🧠"
        st.info(mode_text)
    
    # Auto-refresh functionality
    if st.session_state.get('auto_refresh', True):
        # Refresh feedback stats periodically
        if len(st.session_state.chat_history) > 0:
            # Get latest feedback stats
            latest_feedback_stats = APIClient.get_feedback_stats()
            if latest_feedback_stats:
                st.session_state.feedback_stats = latest_feedback_stats

if __name__ == "__main__":
    main()