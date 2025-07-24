import sqlite3
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from contextlib import contextmanager
import datetime

from graph_models import GraphNode, GraphEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)


class GraphDatabaseInterface:
    """Interface for graph database operations"""
    
    def save_nodes(self, nodes: List[GraphNode]) -> bool:
        raise NotImplementedError
    
    def save_edges(self, edges: List[GraphEdge]) -> bool:
        raise NotImplementedError
    
    def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        raise NotImplementedError
    
    def get_nodes_by_chunk_id(self, chunk_id: str) -> List[GraphNode]:
        raise NotImplementedError
    
    def get_connected_nodes(self, node_id: str, max_depth: int = 2) -> List[GraphNode]:
        raise NotImplementedError
    
    def get_edges_between_nodes(self, node_ids: List[str]) -> List[GraphEdge]:
        raise NotImplementedError
    
    def search_nodes_by_name(self, search_term: str, limit: int = 10) -> List[GraphNode]:
        raise NotImplementedError
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        raise NotImplementedError


class SQLiteGraphDatabase(GraphDatabaseInterface):
    """SQLite implementation of graph database"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            # Nodes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    description TEXT,
                    attributes TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            # Edges table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    description TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY (source_node_id) REFERENCES nodes (id),
                    FOREIGN KEY (target_node_id) REFERENCES nodes (id)
                )
            """)
            
            # Node-chunk mapping table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS node_chunks (
                    node_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    PRIMARY KEY (node_id, chunk_id),
                    FOREIGN KEY (node_id) REFERENCES nodes (id)
                )
            """)
            
            # Edge-chunk mapping table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_chunks (
                    edge_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    PRIMARY KEY (edge_id, chunk_id),
                    FOREIGN KEY (edge_id) REFERENCES edges (id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes (name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes (type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges (source_node_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target_node_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_node_chunks_chunk ON node_chunks (chunk_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_chunks_chunk ON edge_chunks (chunk_id)")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper handling"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def save_nodes(self, nodes: List[GraphNode]) -> bool:
        """Save nodes to database"""
        try:
            with self._get_connection() as conn:
                for node in nodes:
                    # Fix: Handle enum conversion properly
                    node_type_str = node.type.value if hasattr(node.type, 'value') else str(node.type)
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO nodes 
                        (id, name, type, description, attributes, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        node.id,
                        node.name,
                        node_type_str,  # Use the string value
                        node.description,
                        json.dumps(node.attributes) if node.attributes else None,
                        node.created_at.isoformat() if hasattr(node.created_at, 'isoformat') else str(node.created_at),
                        node.updated_at.isoformat() if hasattr(node.updated_at, 'isoformat') else str(node.updated_at)
                    ))
                    
                    # Save chunk associations
                    # First, delete existing associations for this node
                    conn.execute("DELETE FROM node_chunks WHERE node_id = ?", (node.id,))
                    
                    # Insert new associations
                    for chunk_id in node.chunk_ids:
                        conn.execute("""
                            INSERT INTO node_chunks (node_id, chunk_id)
                            VALUES (?, ?)
                        """, (node.id, chunk_id))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save nodes: {e}")
            return False
        
    def save_edges(self, edges: List[GraphEdge]) -> bool:
        """Save edges to database"""
        try:
            with self._get_connection() as conn:
                for edge in edges:
                    # Fix: Handle enum conversion properly
                    edge_type_str = edge.edge_type.value if hasattr(edge.edge_type, 'value') else str(edge.edge_type)
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO edges 
                        (id, source_node_id, target_node_id, edge_type, weight, description, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        edge.id,
                        edge.source_node_id,
                        edge.target_node_id,
                        edge_type_str,  # Use the string value
                        edge.weight,
                        edge.description,
                        edge.created_at.isoformat() if hasattr(edge.created_at, 'isoformat') else str(edge.created_at)
                    ))
                    
                    # Save chunk associations
                    # First, delete existing associations for this edge
                    conn.execute("DELETE FROM edge_chunks WHERE edge_id = ?", (edge.id,))
                    
                    # Insert new associations
                    for chunk_id in edge.chunk_ids:
                        conn.execute("""
                            INSERT INTO edge_chunks (edge_id, chunk_id)
                            VALUES (?, ?)
                        """, (edge.id, chunk_id))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save edges: {e}")
            return False
    
    def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID"""
        try:
            with self._get_connection() as conn:
                row = conn.execute("""
                    SELECT id, name, type, description, attributes, created_at, updated_at
                    FROM nodes WHERE id = ?
                """, (node_id,)).fetchone()
                
                if row:
                    # Get chunk IDs for this node
                    chunk_rows = conn.execute("""
                        SELECT chunk_id FROM node_chunks WHERE node_id = ?
                    """, (node_id,)).fetchall()
                    
                    chunk_ids = {chunk_row[0] for chunk_row in chunk_rows}
                    
                    # Fix: Handle enum conversion from database
                    try:
                        node_type = NodeType(row['type'])
                    except ValueError:
                        logger.warning(f"Invalid node type '{row['type']}' for node {node_id}, using 'concept'")
                        node_type = NodeType.CONCEPT
                    
                    return GraphNode(
                        id=row['id'],
                        name=row['name'],
                        type=node_type,
                        description=row['description'],
                        attributes=json.loads(row['attributes']) if row['attributes'] else {},
                        chunk_ids=chunk_ids,
                        created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                        updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else datetime.now()
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None
    
    def get_nodes_by_chunk_id(self, chunk_id: str) -> List[GraphNode]:
        """Get all nodes associated with a chunk"""
        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT n.id, n.name, n.type, n.description, n.attributes, n.created_at, n.updated_at
                    FROM nodes n
                    JOIN node_chunks nc ON n.id = nc.node_id
                    WHERE nc.chunk_id = ?
                """, (chunk_id,)).fetchall()
                
                nodes = []
                for row in rows:
                    # Get all chunk IDs for this node
                    chunk_rows = conn.execute("""
                        SELECT chunk_id FROM node_chunks WHERE node_id = ?
                    """, (row['id'],)).fetchall()
                    
                    chunk_ids = {chunk_row[0] for chunk_row in chunk_rows}
                    
                    node = GraphNode(
                        id=row['id'],
                        name=row['name'],
                        type=NodeType(row['type']),
                        description=row['description'],
                        attributes=json.loads(row['attributes']) if row['attributes'] else {},
                        chunk_ids=chunk_ids,
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                    nodes.append(node)
                
                return nodes
                
        except Exception as e:
            logger.error(f"Failed to get nodes for chunk {chunk_id}: {e}")
            return []
    
    def get_connected_nodes(self, node_id: str, max_depth: int = 2) -> List[GraphNode]:
        """Get nodes connected to a given node within max_depth"""
        try:
            visited = set()
            to_visit = [(node_id, 0)]
            connected_nodes = []
            
            with self._get_connection() as conn:
                while to_visit:
                    current_id, depth = to_visit.pop(0)
                    
                    if current_id in visited or depth > max_depth:
                        continue
                    
                    visited.add(current_id)
                    
                    # Get current node
                    node = self.get_node_by_id(current_id)
                    if node:
                        connected_nodes.append(node)
                    
                    if depth < max_depth:
                        # Get connected nodes
                        connected_rows = conn.execute("""
                            SELECT target_node_id FROM edges WHERE source_node_id = ?
                            UNION
                            SELECT source_node_id FROM edges WHERE target_node_id = ?
                        """, (current_id, current_id)).fetchall()
                        
                        for row in connected_rows:
                            connected_id = row[0]
                            if connected_id not in visited:
                                to_visit.append((connected_id, depth + 1))
            
            return connected_nodes
            
        except Exception as e:
            logger.error(f"Failed to get connected nodes for {node_id}: {e}")
            return []
    
    def get_edges_between_nodes(self, node_ids: List[str]) -> List[GraphEdge]:
        """Get edges between specific nodes"""
        try:
            if not node_ids:
                return []
            
            placeholders = ','.join(['?' for _ in node_ids])
            with self._get_connection() as conn:
                rows = conn.execute(f"""
                    SELECT id, source_node_id, target_node_id, edge_type, weight, description, created_at
                    FROM edges 
                    WHERE source_node_id IN ({placeholders}) AND target_node_id IN ({placeholders})
                """, node_ids + node_ids).fetchall()
                
                edges = []
                for row in rows:
                    # Get chunk IDs for this edge
                    chunk_rows = conn.execute("""
                        SELECT chunk_id FROM edge_chunks WHERE edge_id = ?
                    """, (row['id'],)).fetchall()
                    
                    chunk_ids = {chunk_row[0] for chunk_row in chunk_rows}
                    
                    # Fix: Handle enum conversion from database
                    try:
                        edge_type = EdgeType(row['edge_type'])
                    except ValueError:
                        logger.warning(f"Invalid edge type '{row['edge_type']}' for edge {row['id']}, using 'relates_to'")
                        edge_type = EdgeType.RELATES_TO
                    
                    edge = GraphEdge(
                        id=row['id'],
                        source_node_id=row['source_node_id'],
                        target_node_id=row['target_node_id'],
                        edge_type=edge_type,
                        weight=row['weight'],
                        description=row['description'],
                        chunk_ids=chunk_ids,
                        created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now()
                    )
                    edges.append(edge)
                
                return edges
                
        except Exception as e:
            logger.error(f"Failed to get edges between nodes: {e}")
            return []

    def search_nodes_by_name(self, search_term: str, limit: int = 10) -> List[GraphNode]:
        """Search nodes by name with fuzzy matching"""
        try:
            with self._get_connection() as conn:
                # Use LIKE for partial matching
                rows = conn.execute("""
                    SELECT id, name, type, description, attributes, created_at, updated_at
                    FROM nodes 
                    WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ?
                    ORDER BY 
                        CASE 
                            WHEN LOWER(name) = LOWER(?) THEN 1
                            WHEN LOWER(name) LIKE ? THEN 2
                            ELSE 3
                        END
                    LIMIT ?
                """, (f'%{search_term.lower()}%', f'%{search_term.lower()}%', 
                    search_term.lower(), f'{search_term.lower()}%', limit)).fetchall()
                
                nodes = []
                for row in rows:
                    # Get chunk IDs for this node
                    chunk_rows = conn.execute("""
                        SELECT chunk_id FROM node_chunks WHERE node_id = ?
                    """, (row['id'],)).fetchall()
                    
                    chunk_ids = {chunk_row[0] for chunk_row in chunk_rows}
                    
                    node = GraphNode(
                        id=row['id'],
                        name=row['name'],
                        type=NodeType(row['type']),
                        description=row['description'],
                        attributes=json.loads(row['attributes']) if row['attributes'] else {},
                        chunk_ids=chunk_ids,
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                    nodes.append(node)
                
                return nodes
                
        except Exception as e:
            logger.error(f"Failed to search nodes: {e}")
            return []
        
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the graph"""
        try:
            with self._get_connection() as conn:
                # Count nodes by type
                node_type_counts = {}
                type_rows = conn.execute("""
                    SELECT type, COUNT(*) as count FROM nodes GROUP BY type
                """).fetchall()
                
                for row in type_rows:
                    node_type_counts[row['type']] = row['count']
                
                # Count edges by type
                edge_type_counts = {}
                edge_rows = conn.execute("""
                    SELECT edge_type, COUNT(*) as count FROM edges GROUP BY edge_type
                """).fetchall()
                
                for row in edge_rows:
                    edge_type_counts[row['edge_type']] = row['count']
                
                # Total counts
                total_nodes = conn.execute("SELECT COUNT(*) as count FROM nodes").fetchone()['count']
                total_edges = conn.execute("SELECT COUNT(*) as count FROM edges").fetchone()['count']
                
                return {
                    "total_nodes": total_nodes,
                    "total_edges": total_edges,
                    "node_types": node_type_counts,
                    "edge_types": edge_type_counts
                }
                
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {}