"""
Visualization tools for comparing different memory systems.
Provides various plots and charts for performance analysis.
"""

from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

from .interfaces import MemoryMetrics
from .evaluator import MemoryEvaluator


class MemoryVisualizer:
    """
    Provides visualization tools for comparing memory system performance.
    """
    
    def __init__(self):
        """Initialize the visualizer with default style settings."""
        plt.style.use('seaborn')
        self.colors = sns.color_palette("husl", 8)
    
    def plot_metrics_comparison(self, 
                              metrics_dict: Dict[str, MemoryMetrics],
                              output_path: str = None):
        """
        Create a radar plot comparing different memory systems.
        
        Args:
            metrics_dict: Dictionary mapping system names to their metrics
            output_path: Optional path to save the plot
        """
        # Prepare data
        metrics = ['Retrieval Accuracy', 'Retrieval Speed', 
                  'Memory Efficiency', 'Relevance', 'Consistency']
        
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for idx, (system_name, system_metrics) in enumerate(metrics_dict.items()):
            # Normalize metrics to 0-1 scale
            values = [
                system_metrics.retrieval_accuracy,
                1.0 - min(system_metrics.retrieval_latency / 1000, 1.0),  # Convert to speed
                1.0 - min(system_metrics.memory_utilization / 1e6, 1.0),  # Normalize to MB
                system_metrics.relevance_score,
                system_metrics.consistency_score
            ]
            values = np.concatenate((values, [values[0]]))  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=system_name, color=self.colors[idx])
            ax.fill(angles, values, alpha=0.25, color=self.colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        if output_path:
            plt.savefig(output_path)
        plt.close()
    
    def plot_latency_distribution(self,
                                latencies: Dict[str, List[float]],
                                output_path: str = None):
        """
        Create violin plots showing latency distributions.
        
        Args:
            latencies: Dictionary mapping system names to lists of latency measurements
            output_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        data = []
        names = []
        for system_name, system_latencies in latencies.items():
            data.extend(system_latencies)
            names.extend([system_name] * len(system_latencies))
        
        df = pd.DataFrame({
            'System': names,
            'Latency (ms)': data
        })
        
        sns.violinplot(data=df, x='System', y='Latency (ms)')
        plt.xticks(rotation=45)
        plt.title('Memory Retrieval Latency Distribution')
        
        if output_path:
            plt.savefig(output_path)
        plt.close()
    
    def plot_accuracy_over_time(self,
                              accuracy_data: Dict[str, List[Tuple[datetime, float]]],
                              output_path: str = None):
        """
        Plot accuracy trends over time.
        
        Args:
            accuracy_data: Dictionary mapping system names to lists of (timestamp, accuracy) tuples
            output_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        for idx, (system_name, measurements) in enumerate(accuracy_data.items()):
            timestamps, accuracies = zip(*measurements)
            timestamps = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]
            plt.plot(timestamps, accuracies, 'o-', 
                    label=system_name, color=self.colors[idx])
        
        plt.xlabel('Time (hours)')
        plt.ylabel('Retrieval Accuracy')
        plt.title('Memory System Accuracy Over Time')
        plt.legend()
        plt.grid(True)
        
        if output_path:
            plt.savefig(output_path)
        plt.close()
    
    def plot_memory_growth(self,
                         memory_data: Dict[str, List[Tuple[int, float]]],
                         output_path: str = None):
        """
        Plot memory usage growth with number of entries.
        
        Args:
            memory_data: Dictionary mapping system names to lists of (num_entries, memory_mb) tuples
            output_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        for idx, (system_name, measurements) in enumerate(memory_data.items()):
            num_entries, memory_usage = zip(*measurements)
            plt.plot(num_entries, memory_usage, 'o-', 
                    label=system_name, color=self.colors[idx])
        
        plt.xlabel('Number of Memories')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory System Storage Growth')
        plt.legend()
        plt.grid(True)
        
        if output_path:
            plt.savefig(output_path)
        plt.close()
