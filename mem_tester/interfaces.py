"""
Base interfaces for AI memory systems.
Defines the contract that all memory system adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel


class MemoryEntry(BaseModel):
    """
    Represents a single memory entry in the system.
    """
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}
    importance_score: float = 0.0
    embedding: Optional[List[float]] = None


class BaseMemorySystem(ABC):
    """
    Abstract base class for all memory system implementations.
    All memory system adapters must inherit from this class.
    """
    
    @abstractmethod
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new memory to the system.
        
        Args:
            content: The content of the memory
            metadata: Optional metadata associated with the memory
            
        Returns:
            str: Unique identifier for the stored memory
        """
        pass
    
    @abstractmethod
    def retrieve_memories(self, query: str, k: int = 5) -> List[MemoryEntry]:
        """
        Retrieve relevant memories based on a query.
        
        Args:
            query: The search query
            k: Number of memories to retrieve
            
        Returns:
            List[MemoryEntry]: List of relevant memories
        """
        pass
    
    @abstractmethod
    def update_memory(self, memory_id: str, content: str) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: Unique identifier of the memory to update
            content: New content for the memory
            
        Returns:
            bool: Success status of the update operation
        """
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from the system.
        
        Args:
            memory_id: Unique identifier of the memory to delete
            
        Returns:
            bool: Success status of the deletion operation
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all memories from the system.
        """
        pass


class MemoryMetrics(BaseModel):
    """
    Metrics for evaluating memory system performance.
    """
    retrieval_accuracy: float
    retrieval_latency: float  # in milliseconds
    memory_utilization: float  # in bytes
    relevance_score: float
    consistency_score: float
    custom_metrics: Dict[str, float] = {}
