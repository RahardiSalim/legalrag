import os
import httpx
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import google.generativeai as genai
from typing import List, Optional, Any
from pydantic import Field, PrivateAttr

import os
os.environ["GEMINI_API_KEY"] = "AIzaSyAX9QYngTF_XC2QIOQ1-oJc-Ic10yPxd-E"

from pyvis.network import Network
import networkx as nx

def visualize_graph_documents_pyvis(graph_documents, filename="graph_visualization.html"):
    """
    Visualize graph documents using pyvis
    
    Parameters:
    graph_documents: List of graph documents from transformer
    filename: Output HTML filename
    """
    if not graph_documents:
        print("No graph documents to visualize")
        return
    
    # Create pyvis network
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    
    # Process the first graph document
    graph_doc = graph_documents[0]
    
    # Add nodes with different colors based on type
    node_colors = {
        'Person': '#ff6b6b',
        'Organization': '#4ecdc4', 
        'Location': '#45b7d1',
        'Event': '#96ceb4',
        'Concept': '#feca57',
        'Document': '#ff9ff3',
        'Default': '#c7ecee'
    }
    
    added_nodes = set()
    
    # Add nodes
    for node in graph_doc.nodes:
        if node.id not in added_nodes:
            color = node_colors.get(node.type, node_colors['Default'])
            
            # Create hover tooltip with node properties
            title = f"ID: {node.id}\nType: {node.type}"
            if hasattr(node, 'properties') and node.properties:
                title += f"\nProperties: {node.properties}"
            
            net.add_node(
                node.id,
                label=node.id,
                title=title,
                color=color,
                size=25
            )
            added_nodes.add(node.id)
    
    # Add relationships/edges
    for rel in graph_doc.relationships:
        # Create edge tooltip
        edge_title = f"Relationship: {rel.type}"
        if hasattr(rel, 'properties') and rel.properties:
            edge_title += f"\nProperties: {rel.properties}"
        
        net.add_edge(
            rel.source.id,
            rel.target.id,
            label=rel.type,
            title=edge_title,
            color={'color': '#848484'},
            width=2
        )
    
    # Configure physics for better visualization
    net.barnes_hut()
    
    # Save to HTML
    net.save_graph(filename)
    print(f"Graph visualization saved to {filename}")
    
    # Print summary
    print(f"\nVisualization Summary:")
    print(f"- Nodes: {len(graph_doc.nodes)}")
    print(f"- Relationships: {len(graph_doc.relationships)}")
    print(f"- Node types: {set(node.type for node in graph_doc.nodes)}")
    print(f"- Relationship types: {set(rel.type for rel in graph_doc.relationships)}")
    
    return filename

def create_networkx_from_graph_documents(graph_documents):
    """
    Convert graph documents to NetworkX graph for additional analysis
    
    Parameters:
    graph_documents: List of graph documents from transformer
    
    Returns:
    NetworkX graph object
    """
    if not graph_documents:
        return None
    
    G = nx.Graph()
    graph_doc = graph_documents[0]
    
    # Add nodes with attributes
    for node in graph_doc.nodes:
        G.add_node(node.id, type=node.type, 
                  properties=getattr(node, 'properties', {}))
    
    # Add edges with attributes
    for rel in graph_doc.relationships:
        G.add_edge(rel.source.id, rel.target.id, 
                  relationship_type=rel.type,
                  properties=getattr(rel, 'properties', {}))
    
    return G

class GeminiChat(BaseChatModel):
    """Custom Gemini LLM wrapper for LangChain."""
    
    # Declare fields as Pydantic fields
    model_name: str = Field(default="gemini-pro")
    api_key: Optional[str] = Field(default=None)
    
    # Use PrivateAttr for private fields
    _model: Any = PrivateAttr(default=None)
    
    def __init__(self, model_name: str = "gemini-pro", api_key: Optional[str] = None, **kwargs):
        # Initialize with field values
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        
        # Configure the Gemini API
        actual_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not actual_api_key:
            raise ValueError("GEMINI_API_KEY must be provided")
            
        genai.configure(api_key=actual_api_key)
        
        # Set the private model field
        self._model = genai.GenerativeModel(model_name)
    
    @property
    def model(self):
        """Access the Gemini model."""
        return self._model

    def invoke(self, input_data, config=None, **kwargs):
        """Fixed invoke method to accept config parameter"""
        # Handle different input types
        if isinstance(input_data, dict):
            # If input is a dict (like {"input": "text"}), extract the text
            prompt = input_data.get("input", str(input_data))
        elif isinstance(input_data, str):
            prompt = input_data
        elif isinstance(input_data, list):
            # If prompt is a list of LangChain messages, convert to string
            prompt = "\n".join([m.content for m in input_data if isinstance(m, (HumanMessage, AIMessage))])
        else:
            prompt = str(input_data)
        
        response = self.model.generate_content(prompt)
        return AIMessage(content=response.text)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response from messages."""
        # Convert messages to a single prompt string
        prompt = "\n".join([
            m.content for m in messages 
            if isinstance(m, (HumanMessage, AIMessage)) and hasattr(m, 'content')
        ])
        
        try:
            response = self.model.generate_content(prompt)
            message = AIMessage(content=response.text)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            # Handle API errors gracefully
            error_message = AIMessage(content=f"Error generating response: {str(e)}")
            generation = ChatGeneration(message=error_message)
            return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "gemini-chat"

    @property
    def _identifying_params(self) -> dict:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


# Initialize Gemini LLM
llm = GeminiChat(model_name="gemini-2.5-flash")

# Graph transformer
transformer = LLMGraphTransformer(llm=llm)

# Sample document
text = """

"""

document = Document(page_content=text)

try:
    # Convert to graph
    graph_documents = transformer.convert_to_graph_documents([document])
    
    if not graph_documents or not graph_documents[0].nodes:
        print("No graph data generated. Check your transformer configuration.")
    else:
        # Display
        print("Nodes:")
        for node in graph_documents[0].nodes:
            print(f"  - {node.id}: {node.type}")

        print("\nRelationships:")
        for rel in graph_documents[0].relationships:
            print(f"  - {rel.source.id} --[{rel.type}]--> {rel.target.id}")
        
        # Visualize with pyvis
        visualize_graph_documents_pyvis(graph_documents, "legal_graph_visualization.html")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

#//////////////

def combine_multiple_graph_documents(graph_documents_list):
    """
    Combine multiple graph documents into a single merged graph document
    
    Parameters:
    graph_documents_list: List of lists of graph documents from multiple sources
    
    Returns:
    Single merged graph document
    """
    from langchain_experimental.graph_transformers.llm import GraphDocument
    from langchain_community.graphs.graph_document import Node, Relationship
    
    all_nodes = {}  # Use dict to avoid duplicates by node ID
    all_relationships = []
    node_id_counter = 0
    
    # Process each document's graph
    for graph_documents in graph_documents_list:
        if not graph_documents:
            continue
            
        for graph_doc in graph_documents:
            # Add nodes (avoid duplicates)
            for node in graph_doc.nodes:
                # Create a unique key based on node content
                node_key = f"{node.id}_{node.type}".lower()
                
                if node_key not in all_nodes:
                    all_nodes[node_key] = node
                else:
                    # Merge properties if node already exists
                    existing_node = all_nodes[node_key]
                    if hasattr(node, 'properties') and hasattr(existing_node, 'properties'):
                        existing_node.properties.update(node.properties or {})
            
            # Add all relationships
            all_relationships.extend(graph_doc.relationships)
    
    # Create merged graph document
    merged_nodes = list(all_nodes.values())
    
    # Remove duplicate relationships
    unique_relationships = []
    seen_relationships = set()
    
    for rel in all_relationships:
        rel_key = f"{rel.source.id}_{rel.type}_{rel.target.id}".lower()
        if rel_key not in seen_relationships:
            unique_relationships.append(rel)
            seen_relationships.add(rel_key)
    
    # Create a mock merged graph document
    class MergedGraphDocument:
        def __init__(self, nodes, relationships):
            self.nodes = nodes
            self.relationships = relationships
    
    return MergedGraphDocument(merged_nodes, unique_relationships)

def process_multiple_documents_and_combine(documents, transformer):
    """
    Process multiple documents and combine their graphs
    
    Parameters:
    documents: List of Document objects
    transformer: LLMGraphTransformer instance
    
    Returns:
    Combined graph document
    """
    all_graph_documents = []
    
    print(f"Processing {len(documents)} documents...")
    
    for i, doc in enumerate(documents):
        try:
            print(f"Processing document {i+1}/{len(documents)}")
            graph_docs = transformer.convert_to_graph_documents([doc])
            all_graph_documents.append(graph_docs)
        except Exception as e:
            print(f"Error processing document {i+1}: {e}")
            continue
    
    # Combine all graphs
    combined_graph = combine_multiple_graph_documents(all_graph_documents)
    return combined_graph

def visualize_combined_graph(combined_graph, filename="combined_graph_visualization.html"):
    """
    Visualize the combined graph using pyvis
    """
    if not combined_graph or not combined_graph.nodes:
        print("No combined graph data to visualize")
        return
    
    # Create pyvis network
    net = Network(height="900px", width="100%", bgcolor="#222222", font_color="white")
    
    # Add nodes with different colors based on type
    node_colors = {
        'Person': '#ff6b6b',
        'Organization': '#4ecdc4', 
        'Location': '#45b7d1',
        'Event': '#96ceb4',
        'Concept': '#feca57',
        'Document': '#ff9ff3',
        'Entity': '#a8e6cf',
        'Default': '#c7ecee'
    }
    
    added_nodes = set()
    
    # Add nodes
    for node in combined_graph.nodes:
        if node.id not in added_nodes:
            color = node_colors.get(node.type, node_colors['Default'])
            
            # Create hover tooltip
            title = f"ID: {node.id}\nType: {node.type}"
            if hasattr(node, 'properties') and node.properties:
                title += f"\nProperties: {node.properties}"
            
            # Size nodes based on degree (relationships)
            node_degree = sum(1 for rel in combined_graph.relationships 
                            if rel.source.id == node.id or rel.target.id == node.id)
            node_size = max(15, min(50, 15 + node_degree * 3))
            
            net.add_node(
                node.id,
                label=node.id,
                title=title,
                color=color,
                size=node_size
            )
            added_nodes.add(node.id)
    
    # Add relationships
    for rel in combined_graph.relationships:
        edge_title = f"Relationship: {rel.type}"
        if hasattr(rel, 'properties') and rel.properties:
            edge_title += f"\nProperties: {rel.properties}"
        
        net.add_edge(
            rel.source.id,
            rel.target.id,
            label=rel.type,
            title=edge_title,
            color={'color': '#848484'},
            width=2
        )
    
    # Configure physics
    net.barnes_hut(
        gravity=-80000,
        central_gravity=0.01,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )
    
    # Add control buttons
    net.show_buttons(filter_=['physics'])
    
    # Save to HTML
    net.save_graph(filename)
    print(f"Combined graph visualization saved to {filename}")
    
    # Print summary
    print(f"\nCombined Graph Summary:")
    print(f"- Total Nodes: {len(combined_graph.nodes)}")
    print(f"- Total Relationships: {len(combined_graph.relationships)}")
    print(f"- Node types: {set(node.type for node in combined_graph.nodes)}")
    print(f"- Relationship types: {set(rel.type for rel in combined_graph.relationships)}")
    
    return filename

# Updated main execution code
def main():
    # Initialize Gemini LLM
    llm = GeminiChat(model_name="gemini-1.5-flash")
    transformer = LLMGraphTransformer(llm=llm)
    
    # Create multiple documents
    documents = [
        Document(page_content="""
        Marie Curie, born in Warsaw, Poland, was a physicist and chemist. 
        She pioneered research on radioactivity and was the first woman to win a Nobel Prize. 
        She shared the Nobel Prize in Physics with her husband, Pierre Curie.
        """),
        
        Document(page_content="""
        Pierre Curie was a French physicist who worked with his wife Marie Curie on radioactivity.
        He was born in Paris and studied at the University of Paris.
        Together they discovered the elements polonium and radium.
        """),
        
        Document(page_content="""
        The University of Paris, also known as Sorbonne, is located in France.
        It is one of the oldest universities in Europe, founded in 1150.
        Many Nobel Prize winners have studied or worked there.
        """)
    ]
    
    try:
        # Process and combine multiple documents
        combined_graph = process_multiple_documents_and_combine(documents, transformer)
        
        if combined_graph and combined_graph.nodes:
            # Visualize individual graphs
            for i, doc in enumerate(documents):
                try:
                    graph_docs = transformer.convert_to_graph_documents([doc])
                    if graph_docs and graph_docs[0].nodes:
                        visualize_graph_documents_pyvis(graph_docs, f"individual_graph_{i+1}.html")
                except Exception as e:
                    print(f"Error visualizing individual graph {i+1}: {e}")
            
            # Visualize combined graph
            visualize_combined_graph(combined_graph, "combined_legal_graph.html")
            
            # Display nodes and relationships
            print("\nCombined Graph Nodes:")
            for node in combined_graph.nodes:
                print(f"  - {node.id}: {node.type}")
            
            print("\nCombined Graph Relationships:")
            for rel in combined_graph.relationships:
                print(f"  - {rel.source.id} --[{rel.type}]--> {rel.target.id}")
        else:
            print("No combined graph data generated.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()