"""
Spike Monitor for SNN Training
Provides comprehensive layer-by-layer spike activity monitoring and analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import os
import json

class SpikeMonitor:
    """Monitor and analyze spike activity across SNN layers"""
    
    def __init__(self, max_history=1000, save_dir="spike_logs"):
        """
        Initialize spike monitor
        
        Args:
            max_history: Maximum number of timesteps to keep in history
            save_dir: Directory to save spike analysis plots and logs
        """
        self.max_history = max_history
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Spike activity tracking
        self.spike_history = defaultdict(lambda: deque(maxlen=max_history))
        self.spike_rates = defaultdict(list)
        self.layer_names = []
        
        # Statistics tracking
        self.timestep = 0
        self.epoch_stats = defaultdict(dict)
        self.batch_stats = defaultdict(list)
        
        # Silence detection
        self.silence_threshold = 0.005  # Consider < 0.5% activity as "silent" (relaxed from 1%)
        self.dead_neuron_threshold = 0.001  # Neurons with < 0.1% activity over time
        
    def register_layer(self, layer_name: str):
        """Register a layer for monitoring"""
        if layer_name not in self.layer_names:
            self.layer_names.append(layer_name)
            print(f"🔌 Registered layer for spike monitoring: {layer_name}")
    
    def record_spikes(self, layer_name: str, spike_data: torch.Tensor, timestep: Optional[int] = None):
        """
        Record spike activity for a layer
        
        Args:
            layer_name: Name of the layer
            spike_data: Spike tensor [batch, neurons] or [batch, agents, neurons]
            timestep: Optional timestep identifier
        """
        if timestep is None:
            timestep = self.timestep
            self.timestep += 1
        
        # Convert to numpy and handle different shapes
        if isinstance(spike_data, torch.Tensor):
            spikes = spike_data.detach().cpu().numpy()
        else:
            spikes = np.array(spike_data)

        # Ensure at least 2D: batch x features
        if spikes.ndim == 0:
            spikes = spikes.reshape(1, 1)
        elif spikes.ndim == 1:
            spikes = spikes.reshape(1, -1)
        
        # Flatten to 2D if needed [batch, features]
        if spikes.ndim > 2:
            original_shape = spikes.shape
            spikes = spikes.reshape(spikes.shape[0], -1)
        else:
            original_shape = spikes.shape
        
        # Calculate statistics
        batch_size = spikes.shape[0]
        total_neurons = spikes.shape[1] if spikes.ndim > 1 else 1
        
        # Spike rates per batch
        if spikes.ndim > 1:
            spike_rate = np.mean(spikes > 0, axis=1)  # Rate per batch sample
            overall_rate = np.mean(spikes > 0)  # Overall rate
        else:
            spike_rate = float(spikes > 0)
            overall_rate = spike_rate
        
        # Store in history
        self.spike_history[layer_name].append({
            'timestep': timestep,
            'spike_rate': overall_rate,
            'batch_rates': spike_rate,
            'shape': original_shape,
            'total_neurons': total_neurons,
            'active_neurons': np.sum(spikes > 0),
            'max_activity': np.max(spikes),
            'mean_activity': np.mean(spikes),
            'std_activity': np.std(spikes)
        })
        
        # Update running statistics
        self.spike_rates[layer_name].append(overall_rate)
        
        # Keep only recent history
        if len(self.spike_rates[layer_name]) > self.max_history:
            self.spike_rates[layer_name] = self.spike_rates[layer_name][-self.max_history:]
    
    def detect_silent_layers(self) -> Dict[str, float]:
        """Detect layers with very low spike activity"""
        silent_layers = {}
        
        for layer_name in self.layer_names:
            if layer_name in self.spike_rates and len(self.spike_rates[layer_name]) > 0:
                recent_rate = np.mean(self.spike_rates[layer_name][-10:])  # Last 10 timesteps
                if recent_rate < self.silence_threshold:
                    silent_layers[layer_name] = recent_rate
        
        return silent_layers
    
    def detect_dead_neurons(self, layer_name: str, threshold: Optional[float] = None) -> int:
        """Detect neurons that rarely spike"""
        if threshold is None:
            threshold = self.dead_neuron_threshold
        
        if layer_name not in self.spike_history or len(self.spike_history[layer_name]) == 0:
            return 0
        
        # Get recent spike data
        recent_data = list(self.spike_history[layer_name])[-50:]  # Last 50 timesteps
        if not recent_data:
            return 0
        
        # This is a simplified version - in practice you'd need actual neuron-wise data
        # For now, estimate based on overall activity
        avg_rate = np.mean([d['spike_rate'] for d in recent_data])
        total_neurons = recent_data[-1]['total_neurons']
        
        # Rough estimate of dead neurons
        estimated_dead = int(total_neurons * (1.0 - min(avg_rate / threshold, 1.0)))
        return max(0, estimated_dead)
    
    def get_layer_stats(self, layer_name: str) -> Dict:
        """Get comprehensive statistics for a layer"""
        if layer_name not in self.spike_history or len(self.spike_history[layer_name]) == 0:
            return {}
        
        history = list(self.spike_history[layer_name])
        recent_history = history[-20:]  # Last 20 timesteps
        
        stats = {
            'current_rate': recent_history[-1]['spike_rate'] if recent_history else 0.0,
            'avg_rate_recent': np.mean([h['spike_rate'] for h in recent_history]),
            'avg_rate_overall': np.mean([h['spike_rate'] for h in history]),
            'max_rate': np.max([h['spike_rate'] for h in history]),
            'min_rate': np.min([h['spike_rate'] for h in history]),
            'std_rate': np.std([h['spike_rate'] for h in history]),
            'total_timesteps': len(history),
            'total_neurons': recent_history[-1]['total_neurons'] if recent_history else 0,
            'is_silent': recent_history[-1]['spike_rate'] < self.silence_threshold if recent_history else True,
            'dead_neurons_est': self.detect_dead_neurons(layer_name)
        }
        
        return stats
    
    def print_summary(self, epoch: Optional[int] = None):
        """Print a summary of spike activity across all layers"""
        if epoch is not None:
            print(f"\n🧠 Spike Monitor Summary - Epoch {epoch}")
        else:
            print(f"\n🧠 Spike Monitor Summary - Timestep {self.timestep}")
        
        print("=" * 80)
        
        silent_layers = self.detect_silent_layers()
        
        for layer_name in self.layer_names:
            stats = self.get_layer_stats(layer_name)
            if not stats:
                continue
            
            status_icon = "🔥" if stats['current_rate'] > 0.1 else "⚡" if stats['current_rate'] > 0.01 else "😴"
            silence_warning = " ⚠️ SILENT" if layer_name in silent_layers else ""
            dead_warning = f" 💀 {stats['dead_neurons_est']} dead" if stats['dead_neurons_est'] > 0 else ""
            
            print(f"{status_icon} {layer_name:20s} | "
                  f"Rate: {stats['current_rate']:.3f} "
                  f"(avg: {stats['avg_rate_recent']:.3f}) | "
                  f"Neurons: {stats['total_neurons']:4d} | "
                  f"Range: [{stats['min_rate']:.3f}, {stats['max_rate']:.3f}]"
                  f"{silence_warning}{dead_warning}")
        
        if silent_layers:
            print(f"\n⚠️  WARNING: {len(silent_layers)} silent layers detected!")
            for layer, rate in silent_layers.items():
                print(f"   - {layer}: {rate:.4f} spike rate")
        
        print("=" * 80)
    
    def plot_spike_activity(self, layer_names: Optional[List[str]] = None, save: bool = True):
        """Plot spike activity over time for specified layers"""
        if layer_names is None:
            layer_names = self.layer_names
        
        fig, axes = plt.subplots(len(layer_names), 1, figsize=(12, 3 * len(layer_names)))
        if len(layer_names) == 1:
            axes = [axes]
        
        for i, layer_name in enumerate(layer_names):
            if layer_name not in self.spike_rates or len(self.spike_rates[layer_name]) == 0:
                axes[i].text(0.5, 0.5, f"No data for {layer_name}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{layer_name} - No Data")
                continue
            
            rates = self.spike_rates[layer_name]
            timesteps = range(len(rates))
            
            axes[i].plot(timesteps, rates, 'b-', alpha=0.7, linewidth=1)
            axes[i].axhline(y=self.silence_threshold, color='r', linestyle='--', alpha=0.5, label='Silence threshold')
            axes[i].set_title(f"{layer_name} Spike Activity")
            axes[i].set_ylabel("Spike Rate")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
            # Add recent average
            if len(rates) >= 10:
                recent_avg = np.mean(rates[-10:])
                axes[i].axhline(y=recent_avg, color='g', linestyle='-', alpha=0.7, label=f'Recent avg: {recent_avg:.3f}')
        
        axes[-1].set_xlabel("Timestep")
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, "spike_activity.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Spike activity plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def save_stats(self, epoch: int):
        """Save spike statistics to JSON file"""
        stats_data = {
            'epoch': epoch,
            'timestep': self.timestep,
            'layer_stats': {},
            'silent_layers': self.detect_silent_layers()
        }
        
        for layer_name in self.layer_names:
            stats_data['layer_stats'][layer_name] = self.get_layer_stats(layer_name)
        
        save_path = os.path.join(self.save_dir, f"spike_stats_epoch_{epoch}.json")
        with open(save_path, 'w') as f:
            json.dump(stats_data, f, indent=2, default=str)
        
        print(f"💾 Spike statistics saved to {save_path}")
    
    def reset_epoch(self):
        """Reset counters for new epoch"""
        self.timestep = 0
    
    def get_health_score(self) -> float:
        """Calculate overall network health score (0-1, higher is better)"""
        if not self.layer_names:
            return 0.0
        
        total_score = 0.0
        layer_count = 0
        
        for layer_name in self.layer_names:
            stats = self.get_layer_stats(layer_name)
            if not stats:
                continue
            
            # Score based on activity level (too low or too high is bad)
            rate = stats['current_rate']
            if rate < 0.001:  # Too silent
                layer_score = 0.0
            elif rate > 0.8:  # Too active (might be saturated)
                layer_score = 0.2
            else:
                # Optimal range is 0.01 to 0.3
                if 0.01 <= rate <= 0.3:
                    layer_score = 1.0
                else:
                    layer_score = max(0.1, 1.0 - abs(rate - 0.15) / 0.15)
            
            total_score += layer_score
            layer_count += 1
        
        return total_score / max(layer_count, 1)


def create_spike_monitor(config: Optional[Dict] = None) -> SpikeMonitor:
    """Factory function to create a spike monitor with configuration"""
    if config is None:
        config = {}
    
    return SpikeMonitor(
        max_history=config.get('max_history', 1000),
        save_dir=config.get('save_dir', 'spike_logs')
    )


# Integration helpers for enhanced_train.py
class SpikeTracker:
    """Simple wrapper for easy integration with training loop"""
    
    def __init__(self, monitor: SpikeMonitor):
        self.monitor = monitor
        self.enabled = True
    
    def __call__(self, layer_name: str, data: torch.Tensor):
        """Quick spike recording"""
        if self.enabled and data is not None:
            self.monitor.record_spikes(layer_name, data)
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def summary(self, epoch: int):
        if self.enabled:
            self.monitor.print_summary(epoch)
            return self.monitor.get_health_score()
        return 1.0
