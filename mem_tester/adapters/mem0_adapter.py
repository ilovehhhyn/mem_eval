"""
Adapter for the mem0 memory system.
Implements the BaseMemorySystem interface for mem0.
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from ..interfaces import BaseMemorySystem, MemoryEntry


class Mem0System(BaseMemorySystem):
    """
    Adapter for mem0 memory system.
    Implements vector-based memory storage and retrieval with importance scoring.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the mem0 memory system.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.memories: Dict[str, MemoryEntry] = {}
        self.encoder = SentenceTransformer(model_name)
        self.importance_decay = 0.95  # Decay factor for importance scores
        
    def _compute_embedding(self, text: str) -> List[float]:
        """
        Compute embedding for text using sentence transformer.
        
        Args:
            text: Input text
            
        Returns:
            List[float]: Embedding vector
        """
        return self.encoder.encode(text).tolist()
    
    def _compute_similarity(self, query_embedding: List[float], 
                          memory_embedding: List[float]) -> float:
        """
        Compute cosine similarity between query and memory embeddings.
        
        Args:
            query_embedding: Query embedding vector
            memory_embedding: Memory embedding vector
            
        Returns:
            float: Cosine similarity score
        """
        a = np.array(query_embedding)
        b = np.array(memory_embedding)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _update_importance_scores(self):
        """Update importance scores for all memories using time decay."""
        current_time = datetime.now()
        for memory in self.memories.values():
            time_diff = (current_time - memory.timestamp).total_seconds()
            decay = self.importance_decay ** (time_diff / 86400)  # Decay based on days
            memory.importance_score *= decay
    
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new memory to the system.
        
        Args:
            content: The content of the memory
            metadata: Optional metadata for the memory
            
        Returns:
            str: Unique identifier for the stored memory
        """
        memory_id = str(uuid.uuid4())
        embedding = self._compute_embedding(content)
        
        self.memories[memory_id] = MemoryEntry(
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
            importance_score=1.0,
            embedding=embedding
        )
        
        return memory_id
    
    def retrieve_memories(self, query: str, k: int = 5) -> List[MemoryEntry]:
        """
        Retrieve relevant memories using semantic similarity.
        
        Args:
            query: The search query
            k: Number of memories to retrieve
            
        Returns:
            List[MemoryEntry]: List of relevant memories
        """
        if not self.memories:
            return []
        
        # Update importance scores
        self._update_importance_scores()
        
        # Compute query embedding
        query_embedding = self._compute_embedding(query)
        
        # Score memories by semantic similarity and importance
        scored_memories = []
        for memory in self.memories.values():
            if memory.embedding:
                similarity = self._compute_similarity(query_embedding, memory.embedding)
                # Combined score using similarity and importance
                score = similarity * 0.7 + memory.importance_score * 0.3
                scored_memories.append((score, memory))
        
        # Sort by score and return top k
        scored_memories.sort(key=lambda x: -x[0])
        return [memory for _, memory in scored_memories[:k]]
    
    def update_memory(self, memory_id: str, content: str) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: Unique identifier of the memory to update
            content: New content for the memory
            
        Returns:
            bool: Success status of the update operation
        """
        if memory_id not in self.memories:
            return False
        
        memory = self.memories[memory_id]
        embedding = self._compute_embedding(content)
        
        self.memories[memory_id] = MemoryEntry(
            content=content,
            timestamp=datetime.now(),
            metadata=memory.metadata,
            importance_score=memory.importance_score,
            embedding=embedding
        )
        return True
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from the system.
        
        Args:
            memory_id: Unique identifier of the memory to delete
            
        Returns:
            bool: Success status of the deletion operation
        """
        if memory_id not in self.memories:
            return False
        
        del self.memories[memory_id]
        return True
    
    def clear(self) -> None:
        """
        Clear all memories from the system.
        """
        self.memories.clear()
