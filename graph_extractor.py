import time
import asyncio
from typing import List, Dict, Any, Tuple
import logging
import re
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from config import Config
from services import ModelManager, ServiceException
from graph_models import GraphNode, GraphEdge, NodeType, EdgeType, GraphExtractionResult

logger = logging.getLogger(__name__)


class GraphExtractorInterface:
    """Interface for graph extraction"""
    
    def extract_graph_batch(self, chunks: List[Document], batch_size: int = 5) -> GraphExtractionResult:
        raise NotImplementedError


class LLMGraphExtractor(GraphExtractorInterface):
    """LLM-based graph extraction with batch processing"""
    
    def __init__(self, config: Config, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.api_calls_count = 0
        
        # Prompts for different extraction phases
        self.node_extraction_prompt = PromptTemplate(
            template="""Anda adalah ahli dalam mengekstrak entitas dan konsep dari dokumen hukum Indonesia.

Analisis dokumen-dokumen berikut dan identifikasi:
1. ENTITAS: Nama organisasi, jabatan, produk keuangan, sistem, prosedur spesifik
2. KONSEP: Definisi, prinsip hukum, kategori, klasifikasi
3. REGULASI: Nomor pasal, peraturan, ketentuan khusus
4. TOPIK: Area subjek utama, domain hukum

Dokumen untuk dianalisis:
{documents}

Untuk setiap node yang ditemukan, berikan dalam format JSON:
{{
    "nodes": [
        {{
            "name": "nama_entity",
            "type": "entity|concept|regulation|topic",
            "description": "deskripsi singkat",
            "attributes": {{"key": "value"}},
            "chunk_ids": ["chunk_id1", "chunk_id2"]
        }}
    ]
}}

PENTING: 
- Fokus pada entitas yang BENAR-BENAR DISEBUTKAN dalam teks
- Jangan membuat entitas yang tidak ada
- Gunakan nama persis seperti dalam dokumen
- Sertakan chunk_id untuk setiap entitas

JSON Response:""",
            input_variables=["documents"]
        )
        
        self.edge_extraction_prompt = PromptTemplate(
            template="""Berdasarkan nodes yang telah diidentifikasi, tentukan hubungan antar entitas.

Nodes yang tersedia:
{nodes_info}

Dokumen sumber:
{documents}

Identifikasi hubungan dengan tipe:
- RELATES_TO: berhubungan umum
- DEFINES: mendefinisikan
- REFERENCES: merujuk/mengacu
- CONTAINS: mengandung/berisi
- MODIFIES: memodifikasi/mengubah
- SUPERSEDES: menggantikan

Format JSON:
{{
    "edges": [
        {{
            "source_node_name": "nama_node_sumber",
            "target_node_name": "nama_node_target", 
            "edge_type": "relates_to|defines|references|contains|modifies|supersedes",
            "weight": 0.8,
            "description": "deskripsi hubungan",
            "chunk_ids": ["chunk_id1"]
        }}
    ]
}}

PENTING:
- Hanya buat hubungan yang JELAS disebutkan atau tersirat dalam teks
- Berikan weight berdasarkan kekuatan hubungan (0.1-1.0)
- Sertakan chunk_id sebagai bukti

JSON Response:""",
            input_variables=["nodes_info", "documents"]
        )
    
    def extract_graph_batch(self, chunks: List[Document], batch_size: int = 5) -> GraphExtractionResult:
        """Extract graph from chunks in batches"""
        start_time = time.time()
        all_nodes = []
        all_edges = []
        processed_chunks = []
        
        try:
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                # Extract nodes first
                nodes = self._extract_nodes_batch(batch)
                all_nodes.extend(nodes)
                
                # Rate limiting
                time.sleep(2)
                
                # Extract edges based on found nodes
                if nodes:
                    edges = self._extract_edges_batch(batch, nodes)
                    all_edges.extend(edges)
                
                # Rate limiting
                time.sleep(2)
                
                processed_chunks.extend([chunk.metadata.get('chunk_id', f'chunk_{i}_{j}') 
                                       for j, chunk in enumerate(batch)])
            
            # Deduplicate nodes and edges
            unique_nodes = self._deduplicate_nodes(all_nodes)
            unique_edges = self._deduplicate_edges(all_edges)
            
            processing_time = time.time() - start_time
            
            return GraphExtractionResult(
                nodes=unique_nodes,
                edges=unique_edges,
                processed_chunks=processed_chunks,
                extraction_time=processing_time,
                api_calls_made=self.api_calls_count
            )
            
        except Exception as e:
            logger.error(f"Graph extraction failed: {e}")
            raise ServiceException(f"Graph extraction failed: {e}")
    
    def _extract_nodes_batch(self, chunks: List[Document]) -> List[GraphNode]:
        """Extract nodes from a batch of chunks"""
        try:
            # Prepare documents text
            docs_text = self._prepare_documents_text(chunks)
            
            # Get LLM response
            llm = self.model_manager.get_llm()
            prompt = self.node_extraction_prompt.format(documents=docs_text)
            
            response = llm.invoke(prompt)
            self.api_calls_count += 1
            
            # Parse response
            nodes = self._parse_nodes_response(response.content, chunks)
            
            logger.info(f"Extracted {len(nodes)} nodes from batch")
            return nodes
            
        except Exception as e:
            logger.error(f"Node extraction failed: {e}")
            return []
    
    def _extract_edges_batch(self, chunks: List[Document], nodes: List[GraphNode]) -> List[GraphEdge]:
        """Extract edges from a batch of chunks given the nodes"""
        try:
            if not nodes:
                return []
            
            # Prepare nodes info
            nodes_info = self._prepare_nodes_info(nodes)
            docs_text = self._prepare_documents_text(chunks)
            
            # Get LLM response
            llm = self.model_manager.get_llm()
            prompt = self.edge_extraction_prompt.format(
                nodes_info=nodes_info, 
                documents=docs_text
            )
            
            response = llm.invoke(prompt)
            self.api_calls_count += 1
            
            # Parse response
            edges = self._parse_edges_response(response.content, nodes, chunks)
            
            logger.info(f"Extracted {len(edges)} edges from batch")
            return edges
            
        except Exception as e:
            logger.error(f"Edge extraction failed: {e}")
            return []
    
    def _prepare_documents_text(self, chunks: List[Document]) -> str:
        """Prepare documents text for processing"""
        docs_info = []
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.metadata.get('chunk_id', f'chunk_{i}')
            content = chunk.page_content[:1000]  # Limit content length
            docs_info.append(f"CHUNK_ID: {chunk_id}\nCONTENT: {content}")
        
        return "\n\n---\n\n".join(docs_info)
    
    def _prepare_nodes_info(self, nodes: List[GraphNode]) -> str:
        """Prepare nodes information for edge extraction"""
        nodes_info = []
        for node in nodes:
            info = f"- {node.name} ({node.type.value})"
            if node.description:
                info += f": {node.description}"
            nodes_info.append(info)
        
        return "\n".join(nodes_info)
    
    def _parse_nodes_response(self, response: str, chunks: List[Document]) -> List[GraphNode]:
        """Parse LLM response to extract nodes"""
        import json
        
        try:
            # Clean response
            response = self._clean_json_response(response)
            data = json.loads(response)
            
            nodes = []
            chunk_id_map = {chunk.metadata.get('chunk_id', f'chunk_{i}'): chunk.metadata.get('chunk_id', f'chunk_{i}') 
                           for i, chunk in enumerate(chunks)}
            
            for node_data in data.get('nodes', []):
                try:
                    node = GraphNode(
                        name=node_data.get('name', ''),
                        type=NodeType(node_data.get('type', 'concept')),
                        description=node_data.get('description'),
                        attributes=node_data.get('attributes', {}),
                        chunk_ids=set(node_data.get('chunk_ids', []))
                    )
                    nodes.append(node)
                except Exception as e:
                    logger.warning(f"Failed to parse node: {node_data}, error: {e}")
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to parse nodes response: {e}")
            return []
    
    def _parse_edges_response(self, response: str, nodes: List[GraphNode], chunks: List[Document]) -> List[GraphEdge]:
        """Parse LLM response to extract edges"""
        import json
        
        try:
            # Clean response
            response = self._clean_json_response(response)
            data = json.loads(response)
            
            edges = []
            # Create node name to id mapping
            node_name_to_id = {node.name: node.id for node in nodes}
            
            for edge_data in data.get('edges', []):
                try:
                    source_name = edge_data.get('source_node_name', '')
                    target_name = edge_data.get('target_node_name', '')
                    
                    if source_name in node_name_to_id and target_name in node_name_to_id:
                        edge = GraphEdge(
                            source_node_id=node_name_to_id[source_name],
                            target_node_id=node_name_to_id[target_name],
                            edge_type=EdgeType(edge_data.get('edge_type', 'relates_to')),
                            weight=float(edge_data.get('weight', 1.0)),
                            description=edge_data.get('description'),
                            chunk_ids=set(edge_data.get('chunk_ids', []))
                        )
                        edges.append(edge)
                except Exception as e:
                    logger.warning(f"Failed to parse edge: {edge_data}, error: {e}")
            
            return edges
            
        except Exception as e:
            logger.error(f"Failed to parse edges response: {e}")
            return []
    
    def _clean_json_response(self, response: str) -> str:
        """Clean and extract JSON from LLM response"""
        # Find JSON content between ```json and ``` or just the JSON part
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return response.strip()
    
    def _deduplicate_nodes(self, nodes: List[GraphNode]) -> List[GraphNode]:
        """Remove duplicate nodes based on name and type"""
        seen = set()
        unique_nodes = []
        
        for node in nodes:
            key = (node.name.lower(), node.type)
            if key not in seen:
                seen.add(key)
                unique_nodes.append(node)
            else:
                # Merge chunk_ids if duplicate found
                for existing_node in unique_nodes:
                    if existing_node.name.lower() == node.name.lower() and existing_node.type == node.type:
                        existing_node.chunk_ids.update(node.chunk_ids)
                        break
        
        return unique_nodes
    
    def _deduplicate_edges(self, edges: List[GraphEdge]) -> List[GraphEdge]:
        """Remove duplicate edges"""
        seen = set()
        unique_edges = []
        
        for edge in edges:
            key = (edge.source_node_id, edge.target_node_id, edge.edge_type)
            if key not in seen:
                seen.add(key)
                unique_edges.append(edge)
            else:
                # Merge chunk_ids and update weight for duplicates
                for existing_edge in unique_edges:
                    if (existing_edge.source_node_id == edge.source_node_id and 
                        existing_edge.target_node_id == edge.target_node_id and 
                        existing_edge.edge_type == edge.edge_type):
                        existing_edge.chunk_ids.update(edge.chunk_ids)
                        existing_edge.weight = max(existing_edge.weight, edge.weight)
                        break
        
        return unique_edges