# Graph Visualization Colors
GRAPH_NODE_COLORS = {
    'Person': '#ff6b6b',
    'Organization': '#4ecdc4', 
    'Location': '#45b7d1',
    'Event': '#96ceb4',
    'Concept': '#feca57',
    'Document': '#ff9ff3',
    'Entity': '#a8e6cf',
    'Default': '#c7ecee'
}

# Legal Document Hierarchy Levels
HIERARCHY_LEVELS = {
    'bab': 1, 
    'bagian': 2, 
    'sub_bagian': 3, 
    'paragraf': 4,
    'pasal': 5, 
    'ayat': 6, 
    'huruf': 7, 
    'angka': 8
}

# Legal Document Patterns
LEGAL_PATTERNS = {
    'bab': r'(?i)^BAB\s+([IVXLCDM]+|[0-9]+)[\s\.\-]*(.*)',
    'bagian': r'(?i)^BAGIAN\s+([IVXLCDM]+|[0-9]+)[\s\.\-]*(.*)',
    'pasal': r'(?i)^PASAL\s+([0-9]+)[\s\.\-]*(.*)',
    'ayat': r'(?i)^\(([0-9]+)\)[\s]*(.*)',
    'huruf': r'(?i)^([a-z])\.\s*(.*)',
    'angka': r'(?i)^([0-9]+)\.\s*(.*)',
    'paragraf': r'(?i)^PARAGRAF\s+([0-9]+)[\s\.\-]*(.*)',
    'sub_bagian': r'(?i)^SUB\s+BAGIAN\s+([0-9]+)[\s\.\-]*(.*)'
}

# Text Splitting Separators
TEXT_SEPARATORS = ["\n\n\n", "\n\n", "\n", ".", "?", "!", " ", ""]

# File Type Extensions
SUPPORTED_FILE_TYPES = {
    '.pdf': 'PDF Document',
    '.txt': 'Text Document', 
    '.docx': 'Word Document',
    '.md': 'Markdown Document'
}