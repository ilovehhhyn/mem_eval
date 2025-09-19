"""
Memory System Evaluator

Provides tools and metrics for evaluating different memory system implementations.
"""

import time
from typing import Dict, List, Type
import numpy as np
from datetime import datetime

from .interfaces import BaseMemorySystem, MemoryMetrics, MemoryEntry


class MemoryEvaluator:
    """
    Evaluates memory system implementations against a standard set of metrics.
    """
    
    def __init__(self, memory_system: BaseMemorySystem):
        """
        Initialize the evaluator with a memory system implementation.
        
        Args:
            memory_system: An instance of a memory system implementation
        """
        self.memory_system = memory_system
        self.test_results: Dict[str, MemoryMetrics] = {}
    
    def measure_retrieval_latency(self, queries: List[str], k: int = 5) -> float:
        """
        Measure the average latency of memory retrieval operations.
        
        Args:
            queries: List of test queries
            k: Number of memories to retrieve per query
            
        Returns:
            float: Average latency in milliseconds
        """
        latencies = []
        for query in queries:
            start_time = time.time()
            self.memory_system.retrieve_memories(query, k)
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
        return np.mean(latencies)
    
    def evaluate_retrieval_accuracy(self, test_cases: List[Dict]) -> float:
        """
        Evaluate the accuracy of memory retrieval.
        
        Args:
            test_cases: List of test cases containing queries and expected results
            
        Returns:
            float: Accuracy score between 0 and 1
        """
        correct = 0
        total = len(test_cases)
        
        for test_case in test_cases:
            query = test_case['query']
            expected = set(test_case['expected'])
            retrieved = set([m.content for m in 
                           self.memory_system.retrieve_memories(query)])
            if expected.intersection(retrieved):
                correct += 1
                
        return correct / total if total > 0 else 0.0
    
    def evaluate_consistency(self, operations: List[Dict]) -> float:
        """
        Evaluate the consistency of the memory system across operations.
        
        Args:
            operations: List of test operations to perform
            
        Returns:
            float: Consistency score between 0 and 1
        """
        success_count = 0
        total_ops = len(operations)
        
        for op in operations:
            try:
                if op['type'] == 'add':
                    self.memory_system.add_memory(op['content'])
                    success_count += 1
                elif op['type'] == 'update':
                    success = self.memory_system.update_memory(
                        op['memory_id'], op['content'])
                    if success:
                        success_count += 1
                elif op['type'] == 'delete':
                    success = self.memory_system.delete_memory(op['memory_id'])
                    if success:
                        success_count += 1
            except Exception:
                continue
                
        return success_count / total_ops if total_ops > 0 else 0.0
    
    def run_evaluation(self, test_suite: Dict) -> MemoryMetrics:
        """
        Run a complete evaluation using a test suite.
        
        Args:
            test_suite: Dictionary containing test cases and parameters
            
        Returns:
            MemoryMetrics: Computed metrics for the memory system
        """
        # Clear any existing memories
        self.memory_system.clear()
        
        # Run individual evaluations
        retrieval_latency = self.measure_retrieval_latency(
            test_suite['retrieval_queries'])
        retrieval_accuracy = self.evaluate_retrieval_accuracy(
            test_suite['accuracy_tests'])
        consistency_score = self.evaluate_consistency(
            test_suite['consistency_tests'])
        
        # Calculate memory utilization (implementation specific)
        memory_utilization = 0.0  # This should be implemented based on specific needs
        
        # Calculate relevance score (can be customized based on needs)
        relevance_score = retrieval_accuracy * 0.7 + consistency_score * 0.3
        
        return MemoryMetrics(
            retrieval_accuracy=retrieval_accuracy,
            retrieval_latency=retrieval_latency,
            memory_utilization=memory_utilization,
            relevance_score=relevance_score,
            consistency_score=consistency_score
        )
