import time
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from langchain.schema import Document

from config.settings import settings

logger = logging.getLogger(__name__)


class ContextFormatter:
    """Format document context with metadata for RAG responses"""
    
    @staticmethod
    def format_with_metadata(documents: List[Document]) -> str:
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            
            source = ContextFormatter._extract_source_name(metadata.get('source', 'Unknown'))
            page = metadata.get('page', metadata.get('page_label', 'Unknown'))
            section_info = ContextFormatter._extract_section_info(metadata)
            processed_date = ContextFormatter._format_processed_date(metadata.get('processed_at'))
            doc_type = metadata.get('document_type', 'general')
            
            context_header = f"""
                [DOKUMEN {i}]
                SUMBER: {source}
                HALAMAN: {page}
                STRUKTUR: {section_info}
                DIPROSES: {processed_date}
                TIPE: {doc_type.upper()}

                KONTEN:"""
            
            context_parts.extend([context_header, doc.page_content, "=" * 80])
        
        return "\n".join(context_parts)

    @staticmethod
    def _extract_source_name(source_path: str) -> str:
        try:
            source_name = Path(source_path).name
            
            if 'tmp' in source_name.lower():
                parts = source_name.split('\\')
                for part in reversed(parts):
                    if not part.startswith('tmp') and part.endswith('.pdf'):
                        source_name = part
                        break
            
            max_length = settings.memory.max_source_name_length
            if len(source_name) > max_length:
                source_name = source_name[:max_length-3] + settings.memory.source_name_truncate_suffix
                
            return source_name
        except Exception:
            return "Unknown Document"

    @staticmethod
    def _extract_section_info(metadata: Dict) -> str:
        section_parts = []
        
        parent_sections = metadata.get('parent_sections', '')
        if parent_sections:
            section_parts.append(parent_sections)
        
        section_type = metadata.get('section_type', '')
        section_number = metadata.get('section_number', '')
        
        if section_type and section_number:
            current_section = f"{section_type.upper()} {section_number}"
            section_parts.append(current_section)
        
        if not section_parts:
            chunk_id = metadata.get('chunk_id', '')
            if chunk_id:
                section_parts.append(f"Chunk: {chunk_id}")
        
        return " > ".join(section_parts) if section_parts else "General Content"

    @staticmethod
    def _format_processed_date(timestamp) -> str:
        if not timestamp:
            return "Unknown"
        
        try:
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
                return dt.strftime("%d/%m/%Y %H:%M")
            else:
                return str(timestamp)
        except Exception:
            return "Unknown Date"


class QueryEnhancer:
    """Enhance queries using chat history context"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def enhance_query(self, query: str, chat_history: List = None) -> str:
        try:
            from langchain.prompts import PromptTemplate
            
            llm = self.model_manager.get_llm()
            history_context = self._format_chat_history(chat_history)
            
            enhancement_prompt = PromptTemplate(
                template=settings.prompt_templates["query_enhancement_template"],
                input_variables=["query", "chat_history"]
            )
            
            formatted_prompt = enhancement_prompt.format(
                query=query, 
                chat_history=history_context
            )
            response = llm.invoke(formatted_prompt)
            
            enhanced_query = response.content.strip()
            return enhanced_query if enhanced_query else query
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return query

    def _format_chat_history(self, chat_history: List = None) -> str:
        if not chat_history:
            return "Tidak ada riwayat percakapan sebelumnya."

        max_history = settings.memory.max_chat_history_display
        recent_history = chat_history[-max_history:] if len(chat_history) > max_history else chat_history
        
        formatted_history = []
        max_content_length = settings.memory.max_response_content_length
        
        for msg in recent_history:
            role = "Pengguna" if msg.role == "user" else "Asisten"
            content = msg.content[:max_content_length] + "..." if len(msg.content) > max_content_length else msg.content
            formatted_history.append(f"{role}: {content}")
        
        return "\n".join(formatted_history) if formatted_history else "Tidak ada riwayat percakapan sebelumnya."