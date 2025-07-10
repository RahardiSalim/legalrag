import streamlit as st
import requests
import json
from typing import List, Dict, Any
import time

# Configure the page
st.set_page_config(
    page_title="Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'last_chunks' not in st.session_state:
    st.session_state.last_chunks = []


def upload_files(files):
    """Upload files to the API"""
    try:
        files_data = []
        for file in files:
            files_data.append(
                ("files", (file.name, file.getvalue(), file.type))
            )
        
        response = requests.post(f"{API_BASE_URL}/upload", files=files_data)
        return response.json()
    except Exception as e:
        st.error(f"Error uploading files: {str(e)}")
        return None


def send_chat_message(question: str, use_enhanced_query: bool = False):
    """Send chat message to the API"""
    try:
        payload = {
            "question": question,
            "use_enhanced_query": use_enhanced_query
        }
        response = requests.post(f"{API_BASE_URL}/chat", json=payload)
        return response.json()
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
        return None


def get_chat_history():
    """Get chat history from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/history")
        return response.json()
    except Exception as e:
        st.error(f"Error getting chat history: {str(e)}")
        return None


def clear_chat_history():
    """Clear chat history"""
    try:
        response = requests.post(f"{API_BASE_URL}/clear-history")
        return response.json()
    except Exception as e:
        st.error(f"Error clearing history: {str(e)}")
        return None


def get_last_chunks():
    """Get last retrieved chunks"""
    try:
        response = requests.get(f"{API_BASE_URL}/chunks")
        return response.json()
    except Exception as e:
        st.error(f"Error getting chunks: {str(e)}")
        return None


def check_health():
    """Check API health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.json()
    except Exception as e:
        return {"status": "error", "system_initialized": False}


# Main App
def main():
    st.title("⚖️ Legal RAG Assistant")
    st.markdown("**Asisten AI untuk Peraturan Hukum dan OJK**")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Dokumen")
        
        uploaded_files = st.file_uploader(
            "Upload file PDF, TXT, atau ZIP",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'zip']
        )
        
        if st.button("📤 Upload dan Proses", type="primary"):
            if uploaded_files:
                with st.spinner("Memproses dokumen..."):
                    result = upload_files(uploaded_files)
                    if result and result.get('success'):
                        st.success(f"✅ {result['message']}")
                        st.session_state.system_initialized = True
                    else:
                        st.error("❌ Gagal memproses dokumen")
            else:
                st.warning("Pilih file terlebih dahulu!")
        
        st.markdown("---")
        
        # System Status
        health = check_health()
        if health.get('system_initialized'):
            st.success("✅ Sistem Siap")
        else:
            st.warning("⚠️ Upload dokumen terlebih dahulu")
        
        st.markdown("---")
        
        # Settings
        st.header("⚙️ Pengaturan")
        use_enhanced_query = st.checkbox(
            "🔍 Gunakan Enhanced Query",
            help="Meningkatkan kualitas pencarian namun lebih lambat"
        )
        
        st.markdown("---")
        
        # Clear History
        if st.button("🗑️ Clear Chat History", type="secondary"):
            result = clear_chat_history()
            if result:
                st.success("✅ History berhasil dihapus")
                st.session_state.chat_history = []
                st.session_state.last_chunks = []
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat")
        
        # Chat input
        user_input = st.text_input(
            "Tanya tentang peraturan hukum:",
            placeholder="Contoh: Apa itu peraturan OJK tentang fintech?"
        )
        
        if st.button("📤 Kirim", type="primary") and user_input:
            if not health.get('system_initialized'):
                st.error("⚠️ Upload dokumen terlebih dahulu!")
            else:
                with st.spinner("Mencari jawaban..."):
                    response = send_chat_message(user_input, use_enhanced_query)
                    if response and "answer" in response:
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": user_input
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response["answer"]
                        })
                        
                        # Store last chunks
                        st.session_state.last_chunks = response.get("source_documents", [])
                        
                        st.rerun()
        
        # Display chat history
        st.markdown("---")
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"**👤 Anda:** {message['content']}")
                else:
                    st.markdown(f"**🤖 Assistant:** {message['content']}")
                    
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("---")
        else:
            st.info("💬 Mulai conversation dengan menanyakan sesuatu!")
    
    with col2:
        st.header("📊 Informasi")
        
        # Show chunks button
        if st.button("📋 Lihat Chunks Terakhir"):
            chunks = get_last_chunks()
            if chunks:
                st.session_state.last_chunks = chunks
        
        # Display last chunks
        if st.session_state.last_chunks:
            st.subheader("📄 Chunks yang Digunakan:")
            for i, chunk in enumerate(st.session_state.last_chunks):
                with st.expander(f"Chunk {i+1}"):
                    st.text_area(
                        f"Content {i+1}:",
                        chunk["content"][:500] + "..." if len(chunk["content"]) > 500 else chunk["content"],
                        height=100,
                        key=f"chunk_{i}"
                    )
                    if chunk.get("metadata"):
                        st.json(chunk["metadata"])
        else:
            st.info("📋 Chunks akan muncul setelah melakukan query")


if __name__ == "__main__":
    main()