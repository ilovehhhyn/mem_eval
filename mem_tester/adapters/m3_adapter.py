"""
Adapter for the m3-agent memory system from ByteDance Seed.
Implements the BaseMemorySystem interface for m3-agent.
"""

import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from ..interfaces import BaseMemorySystem, MemoryEntry


class M3AgentMemory(BaseMemorySystem):
    """
    Adapter for m3-agent memory system.
    Implements hierarchical memory with working memory, core memory, and long-term memory.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 working_memory_size: int = 5,
                 core_memory_size: int = 20):
        """
        Initialize the m3-agent memory system.
        
        Args:
            model_name: Name of the sentence transformer model to use
            working_memory_size: Size of working memory buffer
            core_memory_size: Size of core memory buffer
        """
        self.encoder = SentenceTransformer(model_name)
        self.working_memory_size = working_memory_size
        self.core_memory_size = core_memory_size
        
        # Initialize memory structures
        self.working_memory: Dict[str, MemoryEntry] = {}
        self.core_memory: Dict[str, MemoryEntry] = {}
        self.long_term_memory: Dict[str, MemoryEntry] = {}
        
        # Memory importance thresholds
        self.core_importance_threshold = 0.7
        self.long_term_threshold = 0.5
        
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
    
    def _consolidate_memories(self):
        """
        Consolidate memories between different memory levels based on importance.
        """
        # Move from working to core memory
        working_items = list(self.working_memory.items())
        for memory_id, memory in working_items:
            if memory.importance_score >= self.core_importance_threshold:
                self.core_memory[memory_id] = memory
                del self.working_memory[memory_id]
        
        # Move from core to long-term memory
        core_items = list(self.core_memory.items())
        for memory_id, memory in core_items:
            if len(self.core_memory) > self.core_memory_size:
                if memory.importance_score >= self.long_term_threshold:
                    self.long_term_memory[memory_id] = memory
                del self.core_memory[memory_id]
    
    def _update_importance_scores(self, query_embedding: List[float]):
        """
        Update importance scores based on relevance to current query.
        
        Args:
            query_embedding: Embedding of the current query
        """
        all_memories = {
            **self.working_memory,
            **self.core_memory,
            **self.long_term_memory
        }
        
        for memory in all_memories.values():
            if memory.embedding:
                # Compute relevance to current query
                relevance = self._compute_similarity(query_embedding, memory.embedding)
                # Update importance score with decay and relevance boost
                time_diff = (datetime.now() - memory.timestamp).total_seconds()
                decay = 0.95 ** (time_diff / 86400)  # Daily decay
                memory.importance_score = memory.importance_score * decay + relevance * 0.3
    
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new memory to the working memory.
        
        Args:
            content: The content of the memory
            metadata: Optional metadata for the memory
            
        Returns:
            str: Unique identifier for the stored memory
        """
        memory_id = str(uuid.uuid4())
        embedding = self._compute_embedding(content)
        
        memory = MemoryEntry(
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
            importance_score=1.0,
            embedding=embedding
        )
        
        # Add to working memory
        self.working_memory[memory_id] = memory
        
        # Consolidate if working memory is full
        if len(self.working_memory) > self.working_memory_size:
            self._consolidate_memories()
            
        return memory_id
    
    def retrieve_memories(self, query: str, k: int = 5) -> List[MemoryEntry]:
        """
        Retrieve relevant memories from all memory levels.
        
        Args:
            query: The search query
            k: Number of memories to retrieve
            
        Returns:
            List[MemoryEntry]: List of relevant memories
        """
        query_embedding = self._compute_embedding(query)
        
        # Update importance scores based on query
        self._update_importance_scores(query_embedding)
        
        # Score memories from all levels
        scored_memories: List[Tuple[float, MemoryEntry]] = []
        
        # Helper function to score memories from a specific memory store
        def score_memories(memories: Dict[str, MemoryEntry], level_boost: float):
            for memory in memories.values():
                if memory.embedding:
                    similarity = self._compute_similarity(query_embedding, memory.embedding)
                    # Combined score using similarity, importance, and level boost
                    score = (similarity * 0.5 + 
                            memory.importance_score * 0.3 + 
                            level_boost * 0.2)
                    scored_memories.append((score, memory))
        
        # Score memories with level-specific boosts
        score_memories(self.working_memory, 1.0)  # Working memory gets highest boost
        score_memories(self.core_memory, 0.8)     # Core memory gets medium boost
        score_memories(self.long_term_memory, 0.6) # Long-term memory gets lowest boost
        
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
        # Find memory in any level
        memory = None
        memory_store = None
        
        if memory_id in self.working_memory:
            memory = self.working_memory[memory_id]
            memory_store = self.working_memory
        elif memory_id in self.core_memory:
            memory = self.core_memory[memory_id]
            memory_store = self.core_memory
        elif memory_id in self.long_term_memory:
            memory = self.long_term_memory[memory_id]
            memory_store = self.long_term_memory
            
        if not memory:
            return False
            
        # Update memory
        embedding = self._compute_embedding(content)
        memory_store[memory_id] = MemoryEntry(
            content=content,
            timestamp=datetime.now(),
            metadata=memory.metadata,
            importance_score=memory.importance_score,
            embedding=embedding
        )
        
        return True
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from any memory level.
        
        Args:
            memory_id: Unique identifier of the memory to delete
            
        Returns:
            bool: Success status of the deletion operation
        """
        deleted = False
        
        if memory_id in self.working_memory:
            del self.working_memory[memory_id]
            deleted = True
        elif memory_id in self.core_memory:
            del self.core_memory[memory_id]
            deleted = True
        elif memory_id in self.long_term_memory:
            del self.long_term_memory[memory_id]
            deleted = True
            
        return deleted
    
    def clear(self) -> None:
        """
        Clear all memories from all levels.
        """
        self.working_memory.clear()
        self.core_memory.clear()
        self.long_term_memory.clear()
