"""
Simple in-memory implementation of the memory system interface.
This serves as an example and reference implementation.
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from ..interfaces import BaseMemorySystem, MemoryEntry


class SimpleMemorySystem(BaseMemorySystem):
    """
    A simple in-memory implementation of the memory system interface.
    Uses basic string matching for retrieval.
    """
    
    def __init__(self):
        self.memories: Dict[str, MemoryEntry] = {}
    
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
        self.memories[memory_id] = MemoryEntry(
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
            importance_score=1.0
        )
        return memory_id
    
    def retrieve_memories(self, query: str, k: int = 5) -> List[MemoryEntry]:
        """
        Retrieve relevant memories using simple string matching.
        
        Args:
            query: The search query
            k: Number of memories to retrieve
            
        Returns:
            List[MemoryEntry]: List of relevant memories
        """
        # Simple string matching implementation
        scored_memories = []
        query_terms = query.lower().split()
        
        for memory in self.memories.values():
            score = 0
            content_terms = memory.content.lower().split()
            
            # Calculate simple term overlap score
            for term in query_terms:
                if term in content_terms:
                    score += 1
            
            if score > 0:
                scored_memories.append((score, memory))
        
        # Sort by score and return top k
        scored_memories.sort(key=lambda x: (-x[0], -x[1].importance_score))
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
        self.memories[memory_id] = MemoryEntry(
            content=content,
            timestamp=datetime.now(),
            metadata=memory.metadata,
            importance_score=memory.importance_score
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
