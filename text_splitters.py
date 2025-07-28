import re
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class LegalChunk:
    content: str
    level: int
    section_type: str
    section_number: str
    parent_sections: List[str]
    metadata: Dict[str, Any]

class HierarchicalLegalSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        self.patterns = {
            'bab': r'(?i)^BAB\s+([IVXLCDM]+|[0-9]+)[\s\.\-]*(.*)',
            'bagian': r'(?i)^BAGIAN\s+([IVXLCDM]+|[0-9]+)[\s\.\-]*(.*)',
            'pasal': r'(?i)^PASAL\s+([0-9]+)[\s\.\-]*(.*)',
            'ayat': r'(?i)^\(([0-9]+)\)[\s]*(.*)',
            'huruf': r'(?i)^([a-z])\.\s*(.*)',
            'angka': r'(?i)^([0-9]+)\.\s*(.*)',
            'paragraf': r'(?i)^PARAGRAF\s+([0-9]+)[\s\.\-]*(.*)',
            'sub_bagian': r'(?i)^SUB\s+BAGIAN\s+([0-9]+)[\s\.\-]*(.*)'
        }
        
        self.hierarchy_levels = {
            'bab': 1, 'bagian': 2, 'sub_bagian': 3, 'paragraf': 4,
            'pasal': 5, 'ayat': 6, 'huruf': 7, 'angka': 8
        }
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        all_chunks = []
        for doc in documents:
            legal_chunks = self._parse_legal_structure(doc)
            document_chunks = self._create_hierarchical_chunks(legal_chunks, doc)
            all_chunks.extend(document_chunks)
        return all_chunks
    
    def _parse_legal_structure(self, document: Document) -> List[LegalChunk]:
        lines = document.page_content.split('\n')
        legal_chunks = []
        current_hierarchy = {}
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            matched_pattern = self._find_matching_pattern(line)
            if not matched_pattern:
                continue
                
            pattern_name, section_number, section_title = matched_pattern
            level = self.hierarchy_levels[pattern_name]
            
            current_hierarchy = self._update_hierarchy(current_hierarchy, pattern_name, level, section_number, section_title, line_num)
            parent_sections = self._build_parent_sections(current_hierarchy, pattern_name)
            content_lines = self._collect_content_lines(lines, line_num, line, level)
            
            legal_chunk = LegalChunk(
                content='\n'.join(content_lines),
                level=level,
                section_type=pattern_name,
                section_number=section_number,
                parent_sections=parent_sections,
                metadata=document.metadata.copy()
            )
            legal_chunks.append(legal_chunk)
        
        return legal_chunks
    
    def _find_matching_pattern(self, line: str):
        for pattern_name, pattern in self.patterns.items():
            match = re.match(pattern, line)
            if match:
                section_number = match.group(1)
                section_title = match.group(2) if len(match.groups()) > 1 else ""
                return pattern_name, section_number, section_title
        return None
    
    def _update_hierarchy(self, current_hierarchy, pattern_name, level, section_number, section_title, line_num):
        current_hierarchy = {k: v for k, v in current_hierarchy.items() 
                           if self.hierarchy_levels[k] < level}
        current_hierarchy[pattern_name] = {
            'number': section_number,
            'title': section_title,
            'line_start': line_num
        }
        return current_hierarchy
    
    def _build_parent_sections(self, current_hierarchy, pattern_name):
        parent_sections = []
        for hierarchy_type in sorted(current_hierarchy.keys(), key=lambda x: self.hierarchy_levels[x]):
            if hierarchy_type != pattern_name:
                parent_sections.append(f"{hierarchy_type.upper()} {current_hierarchy[hierarchy_type]['number']}")
        return parent_sections
    
    def _collect_content_lines(self, lines, line_num, line, level):
        content_lines = [line]
        for next_line_num in range(line_num + 1, len(lines)):
            next_line = lines[next_line_num].strip()
            if not next_line:
                content_lines.append("")
                continue
            
            if self._is_higher_section(next_line, level):
                break
            content_lines.append(next_line)
        return content_lines
    
    def _is_higher_section(self, line, current_level):
        for check_pattern, check_regex in self.patterns.items():
            if re.match(check_regex, line):
                check_level = self.hierarchy_levels[check_pattern]
                if check_level <= current_level:
                    return True
        return False
    
    def _create_hierarchical_chunks(self, legal_chunks: List[LegalChunk], original_doc: Document) -> List[Document]:
        chunks = []
        for legal_chunk in legal_chunks:
            if len(legal_chunk.content) > self.chunk_size:
                sub_chunks = self._split_large_chunk(legal_chunk)
                chunks.extend(sub_chunks)
            else:
                chunks.append(self._create_document_chunk(legal_chunk, original_doc))
        return chunks
    
    def _split_large_chunk(self, legal_chunk: LegalChunk) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", ".", "?", "!", " ", ""]
        )
        
        sub_chunks = splitter.split_text(legal_chunk.content)
        documents = []
        
        for i, sub_chunk in enumerate(sub_chunks):
            context_header = self._create_context_header(legal_chunk)
            full_content = f"{context_header}\n\n{sub_chunk}"
            
            metadata = self._create_chunk_metadata(legal_chunk, i, len(sub_chunks), True)
            doc = Document(page_content=full_content, metadata=metadata)
            documents.append(doc)
        
        return documents
    
    def _create_document_chunk(self, legal_chunk: LegalChunk, original_doc: Document) -> Document:
        context_header = self._create_context_header(legal_chunk)
        full_content = f"{context_header}\n\n{legal_chunk.content}"
        metadata = self._create_chunk_metadata(legal_chunk, 0, 1, False)
        return Document(page_content=full_content, metadata=metadata)
    
    def _create_context_header(self, legal_chunk: LegalChunk) -> str:
        context_parts = []
        if legal_chunk.parent_sections:
            context_parts.extend(legal_chunk.parent_sections)
        current_section = f"{legal_chunk.section_type.upper()} {legal_chunk.section_number}"
        context_parts.append(current_section)
        return " > ".join(context_parts)
    
    def _create_chunk_metadata(self, legal_chunk: LegalChunk, sub_chunk_index: int, total_sub_chunks: int, is_sub_chunk: bool):
        metadata = legal_chunk.metadata.copy()
        metadata.update({
            'section_type': legal_chunk.section_type,
            'section_number': legal_chunk.section_number,
            'hierarchy_level': legal_chunk.level,
            'parent_sections': " > ".join(legal_chunk.parent_sections) if legal_chunk.parent_sections else "",
            'sub_chunk_index': sub_chunk_index,
            'total_sub_chunks': total_sub_chunks,
            'is_sub_chunk': is_sub_chunk
        })
        return metadata