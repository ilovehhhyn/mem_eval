# AI Memory Evaluation Framework

A unified testing framework for evaluating different AI memory systems. This framework provides a standardized way to assess and compare various memory implementations such as mem0, memOS, m3-agent, and others. The framework includes built-in support for hierarchical memory systems and advanced memory management strategies.

## Features

- Unified interface for different memory system implementations
- Standardized evaluation metrics
- Extensible adapter system for new memory implementations
- Comprehensive test suite
- Performance and accuracy measurements

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from ai_memory_eval.evaluator import MemoryEvaluator
from ai_memory_eval.adapters.simple_memory import SimpleMemorySystem

# Initialize a memory system
memory_system = SimpleMemorySystem()

# Create an evaluator
evaluator = MemoryEvaluator(memory_system)

# Create a test suite
test_suite = {
    'retrieval_queries': [
        "What is the capital of France?",
        "How does photosynthesis work?"
    ],
    'accuracy_tests': [
        {
            'query': "capital France",
            'expected': ["Paris is the capital of France"]
        }
    ],
    'consistency_tests': [
        {
            'type': 'add',
            'content': "Test memory"
        }
    ]
}

# Run evaluation
metrics = evaluator.run_evaluation(test_suite)

# Access metrics
print(f"Retrieval Accuracy: {metrics.retrieval_accuracy}")
print(f"Retrieval Latency: {metrics.retrieval_latency}ms")
print(f"Consistency Score: {metrics.consistency_score}")
```

### Available Memory Systems

The framework includes built-in support for several memory systems:

1. **SimpleMemory**: A basic in-memory implementation for reference
2. **Mem0**: Vector-based memory system with semantic search
3. **M3Agent**: ByteDance Seed's hierarchical memory system with:
   - Working memory (short-term buffer)
   - Core memory (medium-term storage)
   - Long-term memory
   - Automatic memory consolidation
   - Importance-based memory management
   - Query-based relevance scoring

### Implementing a New Memory System

To add support for a new memory system, create a new adapter that implements the `BaseMemorySystem` interface:

```python
from ai_memory_eval.interfaces import BaseMemorySystem, MemoryEntry

class MyMemorySystem(BaseMemorySystem):
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        # Implementation here
        pass

    def retrieve_memories(self, query: str, k: int = 5) -> List[MemoryEntry]:
        # Implementation here
        pass

    def update_memory(self, memory_id: str, content: str) -> bool:
        # Implementation here
        pass

    def delete_memory(self, memory_id: str) -> bool:
        # Implementation here
        pass

    def clear(self) -> None:
        # Implementation here
        pass
```

## Evaluation Metrics

The framework provides several key metrics:

- **Retrieval Accuracy**: Measures how accurately the system retrieves relevant memories
- **Retrieval Latency**: Measures the average time taken to retrieve memories
- **Memory Utilization**: Tracks the memory usage of the system
- **Consistency Score**: Evaluates the reliability of memory operations
- **Relevance Score**: Combined score for overall system effectiveness

## Running Tests

```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.
