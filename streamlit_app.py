import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the page
st.set_page_config(
    page_title="Legal RAG Assistant",
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
    /* Explicitly set a dark text color for all chat messages */
    color: #1f1f1f; 
    line-height: 1.6;
}
.user-message {
    background-color: #e3f2fd; /* Light blue background */
    border-left: 5px solid #1e88e5; /* A complementary blue for the border */
}
.assistant-message {
    background-color: #e8f5e9; /* Light green background */
    border-left: 5px solid #43a047; /* A complementary green for the border */
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
    background-color: #e0e0e0; /* Slightly darker for better visibility */
    color: #212121; /* Dark text color */
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
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 300
MAX_RETRIES = 3

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'last_chunks' not in st.session_state:
        st.session_state.last_chunks = []
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = None
    if 'system_stats' not in st.session_state:
        st.session_state.system_stats = {}
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False

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
    def upload_files(files) -> Optional[Dict]:
        """Upload files to the API"""
        files_data = []
        for file in files:
            files_data.append(
                ("files", (file.name, file.getvalue(), file.type))
            )
        
        return APIClient._make_request("POST", "/upload", files=files_data)
    
    @staticmethod
    def send_chat_message(question: str, use_enhanced_query: bool = False) -> Optional[Dict]:
        """Send chat message to the API"""
        payload = {
            "question": question,
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
        
        # Upload button
        if st.button("📤 Upload dan Proses", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("Memproses dokumen..."):
                    start_time = time.time()
                    result = APIClient.upload_files(uploaded_files)
                    
                    if result and result.get('success'):
                        processing_time = time.time() - start_time
                        st.success(f"✅ {result['message']}")
                        st.info(f"📊 {result['file_count']} file(s), {result['chunk_count']} chunk(s)")
                        st.info(f"⏱️ Processed in {processing_time:.2f}s")
                        
                        st.session_state.system_initialized = True
                        st.session_state.chat_history = []  # Clear chat history on new upload
                        st.rerun()
                    else:
                        st.error("❌ Gagal memproses dokumen")
            else:
                st.warning("⚠️ Pilih file terlebih dahulu!")
        
        st.markdown("---")
        
        # System Status
        render_system_status()
        
        st.markdown("---")
        
        # Settings
        use_enhanced_query, max_results = render_settings()
        
        st.markdown("---")
        
        # Actions
        render_actions()
        
        return use_enhanced_query, max_results

def render_system_status():
    """Render system status information"""
    st.header("🔧 Status Sistem")
    
    # Get health status
    health = APIClient.check_health()
    
    if health.get('system_initialized'):
        st.success("✅ Sistem Siap")
        st.session_state.system_initialized = True
        
        # Show additional stats if available
        stats = APIClient.get_system_stats()
        if stats:
            st.session_state.system_stats = stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Dokumen", stats.get('document_count', 0))
            with col2:
                st.metric("🔗 Chunks", stats.get('chunk_count', 0))
            
            if stats.get('last_upload_time'):
                st.caption(f"📅 Upload terakhir: {stats['last_upload_time']}")
    else:
        st.warning("⚠️ Upload dokumen terlebih dahulu")
        st.session_state.system_initialized = False
    
    # Health indicators
    with st.expander("🏥 Health Check"):
        st.json(health)

def render_settings():
    """Render settings panel"""
    st.header("⚙️ Pengaturan")
    
    # Query enhancement toggle
    use_enhanced_query = st.checkbox(
        "🔍 Gunakan Enhanced Query",
        value=True,
        help="Menggunakan AI untuk memperbaiki pertanyaan Anda berdasarkan konteks percakapan, membuat pencarian dokumen lebih spesifik dan relevan."
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
        help="Tampilkan informasi debug"
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
    
    return use_enhanced_query, max_results

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
            st.rerun()
    
    # Refresh button
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()

def render_chat_interface(use_enhanced_query: bool):
    """Render the main chat interface"""
    st.header("💬 Chat Assistant")
    
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
                st.form_submit_button("🔍 Debug", use_container_width=True)
    
    # Process chat input
    if submitted and user_input:
        if not st.session_state.system_initialized:
            st.error("⚠️ Upload dokumen terlebih dahulu!")
        else:
            process_chat_message(user_input, use_enhanced_query)
    
    # Display chat history
    display_chat_history()

def process_chat_message(user_input: str, use_enhanced_query: bool):
    """Process a chat message"""
    with st.spinner("🔍 Mencari jawaban..."):
        start_time = time.time()
        response = APIClient.send_chat_message(user_input, use_enhanced_query)
        
        if response and "answer" in response:
            processing_time = time.time() - start_time
            st.session_state.processing_time = processing_time
            
            # Add messages to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"],
                "timestamp": datetime.now(),
                "processing_time": processing_time,
                "enhanced_query": use_enhanced_query,
                "generated_question": response.get("generated_question"),
                "enhanced_query_used": response.get("enhanced_query")
            })
            
            # Store last chunks
            st.session_state.last_chunks = response.get("source_documents", [])
            
            st.success(f"✅ Jawaban ditemukan dalam {processing_time:.2f}s")
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
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 Anda</strong> <small>({timestamp.strftime('%H:%M:%S')})</small><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            
            else:
                processing_info = ""
                if message.get('processing_time'):
                    processing_info = f" ({message['processing_time']:.2f}s"
                    if message.get('enhanced_query'):
                        processing_info += ", Enhanced"
                    processing_info += ")"
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant</strong> <small>({timestamp.strftime('%H:%M:%S')}{processing_info})</small><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show debug info if enabled
                if st.session_state.get('show_debug', False):
                    with st.expander("🐛 Debug Info"):
                        if message.get('generated_question'):
                            st.write(f"**Generated Question:** {message['generated_question']}")
                        if message.get('enhanced_query_used'):
                            st.write(f"**Enhanced Query Used:** {message['enhanced_query_used']}")
            
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("💬 Mulai percakapan dengan menanyakan sesuatu tentang dokumen hukum!")

def render_information_panel(max_results: int):
    """Render the information panel"""
    st.header("📊 Informasi")
    
    # Metrics
    if st.session_state.system_stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💬 Pesan", len(st.session_state.chat_history))
        with col2:
            st.metric("📄 Dokumen", st.session_state.system_stats.get('document_count', 0))
        with col3:
            st.metric("🔗 Chunks", st.session_state.system_stats.get('chunk_count', 0))
    
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

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Title
    st.title("⚖️ Legal RAG Assistant")
    st.markdown("---")
    
    # Render sidebar and get settings
    use_enhanced_query, max_results = render_sidebar()
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_chat_interface(use_enhanced_query)
    
    with col2:
        render_information_panel(max_results)
    
    # Auto-refresh functionality
    if st.session_state.get('auto_refresh', True):
        # Auto refresh every 30 seconds (optional)
        time.sleep(0.1)  

if __name__ == "__main__":
    main()