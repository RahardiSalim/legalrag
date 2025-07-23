from typing import List, Dict, Any, Optional, Set
import logging
import time
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from config import Config
from services import ModelManager, ServiceException
from graph_models import GraphNode, GraphEdge, GraphQueryResult
from graph_database import GraphDatabaseInterface
from graph_extractor import GraphExtractorInterface

logger = logging.getLogger(__name__)


class GraphRAGServiceInterface:
    """Interface for Graph RAG service"""
    
    def build_graph(self, chunks: List[Document]) -> bool:
        raise NotImplementedError
    
    def query_graph(self, question: str, max_nodes: int = 10) -> GraphQueryResult:
        raise NotImplementedError
    
    def get_graph_stats(self) -> Dict[str, Any]:
        raise NotImplementedError


class GraphRAGService(GraphRAGServiceInterface):
    """Graph RAG service implementation"""
    
    def __init__(self, 
                 config: Config,
                 model_manager: ModelManager,
                 graph_database: GraphDatabaseInterface,
                 graph_extractor: GraphExtractorInterface):
        self.config = config
        self.model_manager = model_manager
        self.graph_database = graph_database
        self.graph_extractor = graph_extractor
        
        self.query_prompt = PromptTemplate(
            template="""Anda adalah asisten AI yang ahli dalam menjawab pertanyaan menggunakan knowledge graph.

Berdasarkan informasi dari knowledge graph berikut, jawab pertanyaan dengan akurat:

NODES YANG RELEVAN:
{relevant_nodes}

EDGES YANG RELEVAN:  
{relevant_edges}

PERTANYAAN: {question}

Aturan menjawab:
1. Gunakan informasi dari nodes dan edges untuk memberikan jawaban komprehensif
2. Sebutkan sumber chunk_id jika relevan
3. Jelaskan hubungan antar entitas jika membantu jawaban
4. Jika informasi tidak cukup, katakan dengan jujur
5. Jawab dalam bahasa Indonesia yang formal

JAWABAN:""",
            input_variables=["relevant_nodes", "relevant_edges", "question"]
        )
        
        self.node_search_prompt = PromptTemplate(
            template="""Analisis pertanyaan berikut dan identifikasi entitas kunci yang harus dicari dalam knowledge graph:

PERTANYAAN: {question}

Identifikasi:
1. Entitas utama (organisasi, jabatan, produk, sistem)
2. Konsep hukum yang relevan
3. Topik atau area yang dicari
4. Peraturan atau pasal yang mungkin terkait

Berikan dalam format JSON:
{{
    "search_terms": ["term1", "term2", "term3"],
    "node_types": ["entity", "concept", "regulation", "topic"],
    "priority_terms": ["most_important_term"]
}}

JSON Response:""",
            input_variables=["question"]
        )
    
    def build_graph(self, chunks: List[Document]) -> bool:
        """Build knowledge graph from document chunks"""
        try:
            logger.info(f"Building graph from {len(chunks)} chunks")
            start_time = time.time()
            
            # Extract graph using the extractor
            extraction_result = self.graph_extractor.extract_graph_batch(
                chunks, batch_size=5
            )
            
            # Save nodes to database
            if extraction_result.nodes:
                success_nodes = self.graph_database.save_nodes(extraction_result.nodes)
                if not success_nodes:
                    logger.error("Failed to save nodes to database")
                    return False
                logger.info(f"Saved {len(extraction_result.nodes)} nodes")
            
            # Save edges to database
            if extraction_result.edges:
                success_edges = self.graph_database.save_edges(extraction_result.edges)
                if not success_edges:
                    logger.error("Failed to save edges to database")
                    return False
                logger.info(f"Saved {len(extraction_result.edges)} edges")
            
            processing_time = time.time() - start_time
            logger.info(f"Graph built successfully in {processing_time:.2f}s")
            logger.info(f"API calls made: {extraction_result.api_calls_made}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build graph: {e}")
            raise ServiceException(f"Graph building failed: {e}")
    
    def query_graph(self, question: str, max_nodes: int = 10) -> GraphQueryResult:
        """Query the knowledge graph to answer a question"""
        start_time = time.time()
        
        try:
            # Step 1: Identify relevant search terms from question
            search_terms = self._extract_search_terms(question)
            
            # Step 2: Find relevant nodes based on search terms
            relevant_nodes = self._find_relevant_nodes(search_terms, max_nodes)
            
            if not relevant_nodes:
                return GraphQueryResult(
                    answer="Maaf, saya tidak dapat menemukan informasi yang relevan dalam knowledge graph untuk menjawab pertanyaan tersebut.",
                    relevant_nodes=[],
                    relevant_edges=[],
                    traversal_path=[],
                    source_chunks=set(),
                    processing_time=time.time() - start_time
                )
            
            # Step 3: Get connected nodes for context
            expanded_nodes = self._expand_node_context(relevant_nodes, max_depth=2)
            
            # Step 4: Get relevant edges between nodes
            relevant_edges = self._get_relevant_edges(expanded_nodes)
            
            # Step 5: Generate answer using LLM
            answer = self._generate_graph_answer(question, expanded_nodes, relevant_edges)
            
            # Step 6: Collect source chunks
            source_chunks = set()
            for node in expanded_nodes:
                source_chunks.update(node.chunk_ids)
            for edge in relevant_edges:
                source_chunks.update(edge.chunk_ids)
            
            # Step 7: Create traversal path
            traversal_path = [node.id for node in expanded_nodes[:5]]  # Limit to first 5
            
            processing_time = time.time() - start_time
            
            return GraphQueryResult(
                answer=answer,
                relevant_nodes=expanded_nodes,
                relevant_edges=relevant_edges,
                traversal_path=traversal_path,
                source_chunks=source_chunks,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            raise ServiceException(f"Graph query failed: {e}")
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        try:
            # Get statistics from database
            stats = self.graph_database.get_graph_statistics()
            
            stats.update({
                "status": "active",
                "last_updated": time.time()
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {"status": "error", "message": str(e)}
        
    def _extract_search_terms(self, question: str) -> List[str]:
        """Extract search terms from question using LLM"""
        try:
            llm = self.model_manager.get_llm()
            prompt = self.node_search_prompt.format(question=question)
            
            response = llm.invoke(prompt)
            
            # Parse JSON response
            import json
            import re
            
            # Clean response to extract JSON
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                search_terms = data.get('search_terms', [])
                priority_terms = data.get('priority_terms', [])
                
                # Combine and prioritize terms
                all_terms = priority_terms + [term for term in search_terms if term not in priority_terms]
                return all_terms[:10]  # Limit to 10 terms
            
        except Exception as e:
            logger.warning(f"Failed to extract search terms using LLM: {e}")
        
        # Fallback: simple keyword extraction
        import re
        words = re.findall(r'\b\w+\b', question.lower())
        # Filter out common words
        stopwords = {'apa', 'yang', 'adalah', 'bagaimana', 'mengapa', 'dimana', 'kapan', 
                    'siapa', 'dan', 'atau', 'untuk', 'dalam', 'pada', 'dengan', 'dari'}
        keywords = [word for word in words if len(word) > 3 and word not in stopwords]
        return keywords[:5]
    

    def _find_relevant_nodes(self, search_terms: List[str], max_nodes: int) -> List[GraphNode]:
        """Find nodes relevant to search terms"""
        relevant_nodes = []
        
        try:
            # For each search term, find matching nodes
            for term in search_terms:
                # Search by name similarity (case-insensitive)
                nodes = self._search_nodes_by_name(term.lower())
                relevant_nodes.extend(nodes)
            
            # Remove duplicates and sort by relevance
            unique_nodes = self._deduplicate_and_rank_nodes(relevant_nodes, search_terms)
            
            return unique_nodes[:max_nodes]
            
        except Exception as e:
            logger.error(f"Failed to find relevant nodes: {e}")
            return []
    
    def _deduplicate_and_rank_nodes(self, nodes: List[GraphNode], search_terms: List[str]) -> List[GraphNode]:
        """Remove duplicates and rank nodes by relevance to search terms"""
        unique_nodes = {}
        
        for node in nodes:
            if node.id not in unique_nodes:
                # Calculate relevance score based on how many search terms match
                relevance_score = 0
                node_text = f"{node.name} {node.description or ''}".lower()
                
                for term in search_terms:
                    if term.lower() in node_text:
                        relevance_score += 1
                
                node.attributes['relevance_score'] = relevance_score
                unique_nodes[node.id] = node
        
        # Sort by relevance score
        sorted_nodes = sorted(unique_nodes.values(), 
                            key=lambda x: x.attributes.get('relevance_score', 0), 
                            reverse=True)
        
        return sorted_nodes
            
    def _search_nodes_by_name(self, search_term: str) -> List[GraphNode]:
        """Search for nodes by name using database search"""
        try:
            return self.graph_database.search_nodes_by_name(search_term, limit=10)
        except Exception as e:
            logger.error(f"Failed to search nodes by name: {e}")
            return []
    
    def _expand_node_context(self, nodes: List[GraphNode], max_depth: int = 2) -> List[GraphNode]:
        """Expand context by getting connected nodes"""
        expanded_nodes = list(nodes)  # Start with original nodes
        
        try:
            for node in nodes:
                connected = self.graph_database.get_connected_nodes(node.id, max_depth)
                for connected_node in connected:
                    if connected_node.id not in [n.id for n in expanded_nodes]:
                        expanded_nodes.append(connected_node)
            
            return expanded_nodes
            
        except Exception as e:
            logger.error(f"Failed to expand node context: {e}")
            return nodes
    
    def _get_relevant_edges(self, nodes: List[GraphNode]) -> List[GraphEdge]:
        """Get edges between the given nodes"""
        try:
            node_ids = [node.id for node in nodes]
            edges = self.graph_database.get_edges_between_nodes(node_ids)
            return edges
            
        except Exception as e:
            logger.error(f"Failed to get relevant edges: {e}")
            return []
    
    def _generate_graph_answer(self, question: str, nodes: List[GraphNode], edges: List[GraphEdge]) -> str:
        """Generate answer using graph information"""
        try:
            # Prepare nodes information
            nodes_info = self._format_nodes_for_prompt(nodes)
            edges_info = self._format_edges_for_prompt(edges)
            
            # Generate answer using LLM
            llm = self.model_manager.get_llm()
            prompt = self.query_prompt.format(
                relevant_nodes=nodes_info,
                relevant_edges=edges_info,
                question=question
            )
            
            response = llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate graph answer: {e}")
            return "Maaf, terjadi kesalahan dalam memproses informasi dari knowledge graph."
    
    def _format_nodes_for_prompt(self, nodes: List[GraphNode]) -> str:
        """Format nodes information for LLM prompt"""
        if not nodes:
            return "Tidak ada nodes yang relevan ditemukan."
        
        formatted_nodes = []
        for node in nodes:
            node_info = f"- {node.name} ({node.type.value})"
            if node.description:
                node_info += f": {node.description}"
            if node.chunk_ids:
                chunk_list = list(node.chunk_ids)[:3]  # Limit to 3 chunks
                node_info += f" [Sources: {', '.join(chunk_list)}]"
            formatted_nodes.append(node_info)
        
        return "\n".join(formatted_nodes)
    
    def _format_edges_for_prompt(self, edges: List[GraphEdge]) -> str:
        """Format edges information for LLM prompt"""
        if not edges:
            return "Tidak ada hubungan yang relevan ditemukan."
        
        formatted_edges = []
        for edge in edges:
            edge_info = f"- {edge.source_node_id} --({edge.edge_type.value})--> {edge.target_node_id}"
            if edge.description:
                edge_info += f": {edge.description}"
            if edge.weight < 1.0:
                edge_info += f" (confidence: {edge.weight:.2f})"
            formatted_edges.append(edge_info)
        
        return "\n".join(formatted_edges)