"""
Adapter for mem0.
Implements the BaseMemorySystem interface using the actual mem0 API.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from mem0 import MemoryClient
from ..interfaces import BaseMemorySystem, MemoryEntry


class Mem0System(BaseMemorySystem):
    """
    Adapter for mem0 memory system using the actual mem0 API.
    This properly wraps mem0's LLM-based memory processing.
    """
    
    def __init__(self, api_key: Optional[str] = None, user_id: str = "default_user"):
        """
        Initialize the mem0 memory system.
        
        Args:
            api_key: The mem0 API key (or will use MEM0_API_KEY env var)
            user_id: User identifier for mem0 (required for mem0's API)
        """
        self.client = MemoryClient(api_key=api_key)
        self.user_id = user_id
        # Track memory IDs to support framework operations
        self._memory_id_mapping = {}
        
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new memory using mem0's LLM-based processing.
        
        Args:
            content: The content of the memory
            metadata: Optional metadata for the memory
            
        Returns:
            str: Unique identifier for the stored memory
        """
        # mem0 expects messages format
        messages = [{"role": "user", "content": content}] 
        # kwargs (key word arguments)
        kwargs = {"user_id": self.user_id}
        if metadata:
            kwargs["metadata"] = metadata
        
        # Use mem0's add method 
        response = self.client.add(messages, **kwargs)
        
        # mem0 returns a response with memory info
        # Extract memory ID from response (format may vary according to api)
        if isinstance(response, dict):
            if 'results' in response and response['results']:
                memory_id = response['results'][0].get('id', response['results'][0].get('memory_id'))
            else:
                memory_id = response.get('id', response.get('memory_id'))
        else:
            #  if response format is unexpected
            memory_id = str(response)
        
        return memory_id
    
    def retrieve_memories(self, query: str, k: int = 5) -> List[MemoryEntry]:
        """
        Retrieve relevant memories using mem0's semantic search.
        
        Args:
            query: The search query
            k: Number of memories to retrieve
            
        Returns:
            List[MemoryEntry]: List of relevant memories
        """
        # Use mem0's search method
        response = self.client.search(
            query=query,
            user_id=self.user_id,
            limit=k
        )
        
        # Convert mem0 response to MemoryEntry objects
        memories = []
        if isinstance(response, list):
            results = response
        else:
            results = response.get('results', [])
        
        for item in results:
            # Extract memory data from mem0's response format
            memory_entry = self._convert_mem0_to_memory_entry(item)
            if memory_entry:
                memories.append(memory_entry)
        
        return memories
    
    def update_memory(self, memory_id: str, content: str) -> bool:
        """
        Update an existing memory using mem0's update method.
        
        Args:
            memory_id: Unique identifier of the memory to update
            content: New content for the memory
            
        Returns:
            bool: Success status of update 
        """
        try:
            # Get existing memory to preserve metadata if possible
            existing_metadata = None
            try:
                existing_memory = self.client.get(memory_id=memory_id)
                existing_metadata = existing_memory.get('metadata')
            except:
                pass
            
            # Update with mem0
            update_params = {"text": content}
            if existing_metadata:
                update_params["metadata"] = existing_metadata
                
            response = self.client.update(memory_id=memory_id, **update_params)
            return response.get('message') == 'Memory updated successfully' or 'updated' in str(response).lower()
        except Exception:
            return False
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory using mem0's delete method.
        
        Args:
            memory_id: Unique identifier of the memory to delete
            
        Returns:
            bool: Success status of the deletion operation
        """
        try:
            response = self.client.delete(memory_id=memory_id)
            return response.get('message') == 'Memory deleted successfully' or 'deleted' in str(response).lower()
        except Exception:
            return False
    
    def clear(self) -> None:
        """
        Clear all memories for this user using mem0's delete_all method.
        """
        try:
            self.client.delete_all(user_id=self.user_id)
        except Exception:
            # If delete_all fails, try to get all memories and delete individually
            try:
                all_memories = self.client.get_all(user_id=self.user_id)
                if isinstance(all_memories, list):
                    results = all_memories
                else:
                    results = all_memories.get('results', [])
                
                for memory in results:
                    memory_id = memory.get('id', memory.get('memory_id'))
                    if memory_id:
                        self.client.delete(memory_id=memory_id)
            except Exception:
                pass  # Silent fail for clear operation
    
    def _convert_mem0_to_memory_entry(self, mem0_item: Dict[str, Any]) -> Optional[MemoryEntry]:
        """
        Convert mem0's response format to MemoryEntry according to simplememory framework.
        
        Args:
            mem0_item: Individual memory item from mem0 API response
            
        Returns:
            MemoryEntry: Converted memory entry or None if conversion fails
        """
        try:
            # Extract content - mem0 might use 'memory', 'text', or 'content'
            content = mem0_item.get('memory') or mem0_item.get('text') or mem0_item.get('content', '')
            
            # Extract timestamp - mem0 might use various field names
            timestamp_str = mem0_item.get('created_at') or mem0_item.get('timestamp') or mem0_item.get('updated_at')
            if timestamp_str:
                try:
                    # Try parsing ISO format timestamp
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()
            
            # Extract metadata
            metadata = mem0_item.get('metadata', {})
            
            # Extract or calculate importance score
            importance_score = mem0_item.get('score', mem0_item.get('relevance_score', 1.0))
            
            return MemoryEntry(
                content=content,
                timestamp=timestamp,
                metadata=metadata,
                importance_score=float(importance_score)
            )
        except Exception:
            return None
    
    def get_all_memories(self) -> List[MemoryEntry]:
        """
        Get all memories for the current user (additional helper method).
        
        Returns:
            List[MemoryEntry]: All memories for this user
        """
        try:
            response = self.client.get_all(user_id=self.user_id)
            if isinstance(response, list):
                results = response
            else:
                results = response.get('results', [])
            
            memories = []
            for item in results:
                memory_entry = self._convert_mem0_to_memory_entry(item)
                if memory_entry:
                    memories.append(memory_entry)
            
            return memories
        except Exception:
            return []
