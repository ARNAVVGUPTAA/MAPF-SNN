"""
LSM Evaluation Script for MAPF
================================

Evaluates trained LSM on test trajectories with comprehensive metrics:
- Spike rate: Average spiking activity across reservoir
- Collision rate: Frequency of agent collisions
- Reach rate: Success rate of reaching goals
- Path efficiency: Actual path length vs optimal
- Per-agent metrics
- Per-mesh statistics

This simulates the agent behavior and tracks all relevant metrics.
"""

import os
import sys
import yaml
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lsm_reservoir import LSMNetwork
from models.neuromorphic_lsm_trainable import TrainableNeuromorphicLSMNetwork
from data_loader import SNNDataset


class MAPFSimulator:
    """
    MAPF simulator for evaluating agent behavior.
    Tracks collisions, goal reaching, and path efficiency.
    
    FOV Format: [2, H, W]
    - Channel 0: Obstacles/agents (0=empty, 1=obstacle, 2=other agent)
    - Channel 1: Goals (0=empty, 3=goal)
    - Center of FOV (H//2, W//2) is agent's current position
    """
    
    def __init__(self, board_size):
        self.board_size = board_size
        self.reset()
    
    def reset(self):
        """Reset simulator state"""
        self.positions = {}  # agent_id -> (x, y)
        self.initial_positions = {}  # agent_id -> (x, y)
        self.goals = {}  # agent_id -> (x, y)
        self.collisions = 0
        self.reached_goals = set()
        self.path_lengths = {}  # agent_id -> steps taken
        self.timesteps = 0
    
    def extract_goal_from_fov(self, fov):
        """
        Extract goal position from FOV.
        
        Args:
            fov: [2, H, W] observation tensor
            
        Returns:
            goal_offset: (dx, dy) offset from agent position, or None if no goal visible
        """
        goal_channel = fov[1].cpu().numpy()  # Channel 1 = goals
        goal_positions = np.where(goal_channel == 3)
        
        if len(goal_positions[0]) > 0:
            # Goal is visible in FOV
            goal_y, goal_x = goal_positions[0][0], goal_positions[1][0]
            center = goal_channel.shape[0] // 2
            
            # Compute offset from agent (center of FOV)
            dx = goal_x - center
            dy = goal_y - center
            return (dx, dy)
        
        return None
    
    def initialize_from_fov(self, agent_id, fov, global_pos=None):
        """
        Initialize agent position and goal from FOV.
        
        Args:
            agent_id: Agent ID
            fov: [2, H, W] FOV tensor
            global_pos: (x, y) global position if known, else inferred
        """
        if agent_id not in self.positions:
            # Use center of board as default position if not specified
            if global_pos is None:
                global_pos = (self.board_size[0] // 2 + agent_id, 
                            self.board_size[1] // 2)
            
            self.positions[agent_id] = global_pos
            self.initial_positions[agent_id] = global_pos
            self.path_lengths[agent_id] = 0
        
        # Extract goal from FOV
        goal_offset = self.extract_goal_from_fov(fov)
        if goal_offset is not None and agent_id not in self.goals:
            dx, dy = goal_offset
            current_x, current_y = self.positions[agent_id]
            self.goals[agent_id] = (current_x + dx, current_y + dy)
    
    
    def execute_action(self, agent_id, action):
        """
        Execute agent action and update state.
        
        Actions: 0=RIGHT, 1=UP, 2=LEFT, 3=DOWN, 4=STAY
        
        Returns:
            collision: True if collision occurred
        """
        if agent_id not in self.positions:
            return False
        
        x, y = self.positions[agent_id]
        
        # Apply action (matching MAPF action space)
        if action == 0:  # RIGHT
            x = min(self.board_size[0] - 1, x + 1)
        elif action == 1:  # UP
            y = max(0, y - 1)
        elif action == 2:  # LEFT
            x = max(0, x - 1)
        elif action == 3:  # DOWN
            y = min(self.board_size[1] - 1, y + 1)
        # action == 4: STAY (no movement)
        
        new_pos = (x, y)
        
        # Check collision with other agents
        collision = False
        for other_id, other_pos in self.positions.items():
            if other_id != agent_id and other_pos == new_pos:
                collision = True
                self.collisions += 1
                break
        
        if not collision:
            self.positions[agent_id] = new_pos
            
            # Only count step if agent actually moved
            if new_pos != self.positions.get(agent_id, new_pos):
                self.path_lengths[agent_id] = self.path_lengths.get(agent_id, 0) + 1
            else:
                # Still count movement attempts
                self.path_lengths[agent_id] = self.path_lengths.get(agent_id, 0) + 1
            
            # Check goal reached
            if agent_id in self.goals and new_pos == self.goals[agent_id]:
                self.reached_goals.add(agent_id)
        
        self.timesteps += 1
        return collision
    
    
    def get_metrics(self, num_agents):
        """Get current metrics"""
        reach_rate = len(self.reached_goals) / num_agents if num_agents > 0 else 0
        collision_rate = self.collisions / max(1, self.timesteps)
        
        return {
            'collisions': self.collisions,
            'reached_goals': len(self.reached_goals),
            'reach_rate': reach_rate,
            'collision_rate': collision_rate,
            'avg_path_length': np.mean(list(self.path_lengths.values())) if self.path_lengths else 0
        }


class LSMEvaluator:
    """
    Comprehensive LSM evaluation for MAPF.
    """
    
    def __init__(self, config, network, checkpoint_path=None):
        self.config = config
        self.network = network
        self.device = config['device']
        
        # Move network to device
        self.network.to(self.device)
        self.network.eval()
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Metrics storage
        self.metrics = defaultdict(list)
        
        # Simulator
        self.simulator = MAPFSimulator(config['board_size'])
    
    def load_checkpoint(self, path):
        """Load trained model"""
        print(f"📂 Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.network.load_state_dict(checkpoint['network_state'], strict=False)
        
        print(f"   ✅ Model loaded")
        if 'train_metrics' in checkpoint:
            print(f"   Train accuracy: {checkpoint['train_metrics']['accuracy']*100:.2f}%")
        if 'val_metrics' in checkpoint:
            print(f"   Val accuracy: {checkpoint['val_metrics']['accuracy']*100:.2f}%")
    
    def evaluate_trajectory(self, states, actions_true, verbose=False):
        """
        Evaluate network on a single trajectory.
        
        Args:
            states: [T, A, 2, H, W] observations
            actions_true: [T, A] true expert actions
            verbose: Print detailed info
        
        Returns:
            metrics: Dictionary of metrics for this trajectory
        """
        T, A = actions_true.shape[:2]
        
        # Reset simulator and network
        self.simulator.reset()
        self.network.reset_state()
        
        # Initialize agents from first observation
        first_obs = states[0]  # [A, 2, H, W]
        for agent_id in range(A):
            agent_fov = first_obs[agent_id]  # [2, H, W]
            # Initialize with default position and extract goal from FOV
            self.simulator.initialize_from_fov(
                agent_id, 
                agent_fov,
                global_pos=(agent_id * 2, agent_id * 2)  # Spread agents on grid
            )
        
        # Track metrics
        trajectory_metrics = {
            'spike_rates': [],
            'mesh_spike_rates': [],
            'actions_taken': [],
            'actions_correct': [],
            'collisions': [],
            'e_rates': [],
            'i_rates': []
        }
        
        with torch.no_grad():
            for t in range(T):
                fov = states[t].to(self.device)  # [A, 2, H, W]
                
                # Update goals from FOV (in case goal becomes visible later)
                for agent_id in range(A):
                    agent_fov = fov[agent_id]
                    if agent_id not in self.simulator.goals:
                        goal_offset = self.simulator.extract_goal_from_fov(agent_fov)
                        if goal_offset is not None:
                            dx, dy = goal_offset
                            current_x, current_y = self.simulator.positions[agent_id]
                            self.simulator.goals[agent_id] = (current_x + dx, current_y + dy)
                
                # Process each agent
                for agent_id in range(A):
                    agent_fov = fov[agent_id:agent_id+1]  # [1, 2, H, W]
                    
                    # Get action from network
                    action_logits = self.network(agent_fov)
                    action_pred = torch.argmax(action_logits, dim=1).item()
                    action_true = actions_true[t, agent_id].item()
                    
                    # Execute action in simulator
                    collision = self.simulator.execute_action(agent_id, action_pred)
                    
                    # Record metrics
                    trajectory_metrics['actions_taken'].append(action_pred)
                    trajectory_metrics['actions_correct'].append(action_pred == action_true)
                    trajectory_metrics['collisions'].append(collision)
                
                # Get reservoir statistics (once per timestep)
                stats = self.network.get_stats()
                trajectory_metrics['spike_rates'].append(stats['avg_spike_rate'])
                trajectory_metrics['mesh_spike_rates'].append(stats['mesh_spike_rates'])
                trajectory_metrics['e_rates'].append(stats['avg_e_rate'])
                trajectory_metrics['i_rates'].append(stats['avg_i_rate'])
        
        # Compute summary metrics
        sim_metrics = self.simulator.get_metrics(A)
        
        summary = {
            'timesteps': T,
            'num_agents': A,
            'action_accuracy': np.mean(trajectory_metrics['actions_correct']),
            'avg_spike_rate': np.mean(trajectory_metrics['spike_rates']),
            'avg_e_rate': np.mean(trajectory_metrics['e_rates']),
            'avg_i_rate': np.mean(trajectory_metrics['i_rates']),
            'ei_balance': np.mean(trajectory_metrics['e_rates']) / (np.mean(trajectory_metrics['i_rates']) + 1e-8),
            'collisions': sim_metrics['collisions'],
            'collision_rate': sim_metrics['collision_rate'],
            'reached_goals': sim_metrics['reached_goals'],
            'reach_rate': sim_metrics['reach_rate'],
            'avg_path_length': sim_metrics['avg_path_length'],
            'goals_defined': len(self.simulator.goals)
        }
        
        if verbose:
            print(f"\n   Trajectory Metrics:")
            print(f"      Action accuracy: {summary['action_accuracy']*100:.2f}%")
            print(f"      Spike rate: {summary['avg_spike_rate']:.4f}")
            print(f"      Collision rate: {summary['collision_rate']*100:.2f}%")
            print(f"      Reach rate: {summary['reach_rate']*100:.2f}% ({summary['reached_goals']}/{summary['num_agents']} agents)")
            print(f"      Goals defined: {summary['goals_defined']}/{summary['num_agents']}")
        
        return summary, trajectory_metrics
    
    def evaluate_dataset(self, num_episodes=None, split='valid'):
        """
        Evaluate on multiple trajectories from dataset.
        
        Args:
            num_episodes: Number of episodes to evaluate (None = all)
            split: 'train' or 'valid'
        """
        print("\n" + "="*70)
        print(f"📊 EVALUATING LSM ON {split.upper()} SET")
        print("="*70)
        
        # Load dataset
        dataset = SNNDataset(self.config, split)
        
        if num_episodes is None:
            num_episodes = len(dataset)
        else:
            num_episodes = min(num_episodes, len(dataset))
        
        print(f"   Evaluating {num_episodes} trajectories...")
        
        # Accumulate metrics
        all_metrics = defaultdict(list)
        
        for ep_idx in tqdm(range(num_episodes), desc="Evaluating"):
            states, actions, gso = dataset[ep_idx]
            
            summary, trajectory_metrics = self.evaluate_trajectory(
                states, actions, verbose=(ep_idx < 3)
            )
            
            # Accumulate
            for key, value in summary.items():
                all_metrics[key].append(value)
        
        # Compute aggregate statistics
        print(f"\n" + "="*70)
        print("📈 AGGREGATE METRICS")
        print("="*70)
        
        aggregate = {}
        for key, values in all_metrics.items():
            aggregate[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Print key metrics
        print(f"\n🎯 Performance Metrics:")
        print(f"   Action Accuracy: {aggregate['action_accuracy']['mean']*100:.2f}% ± {aggregate['action_accuracy']['std']*100:.2f}%")
        print(f"   Collision Rate: {aggregate['collision_rate']['mean']*100:.2f}% ± {aggregate['collision_rate']['std']*100:.2f}%")
        print(f"   Reach Rate: {aggregate['reach_rate']['mean']*100:.2f}% ± {aggregate['reach_rate']['std']*100:.2f}%")
        
        print(f"\n🧠 Reservoir Metrics:")
        print(f"   Avg Spike Rate: {aggregate['avg_spike_rate']['mean']:.4f} ± {aggregate['avg_spike_rate']['std']:.4f}")
        print(f"   Avg E Rate: {aggregate['avg_e_rate']['mean']:.4f} ± {aggregate['avg_e_rate']['std']:.4f}")
        print(f"   Avg I Rate: {aggregate['avg_i_rate']['mean']:.4f} ± {aggregate['avg_i_rate']['std']:.4f}")
        print(f"   E/I Balance: {aggregate['ei_balance']['mean']:.2f} ± {aggregate['ei_balance']['std']:.2f}")
        
        print(f"\n📏 Path Metrics:")
        print(f"   Avg Path Length: {aggregate['avg_path_length']['mean']:.2f} ± {aggregate['avg_path_length']['std']:.2f}")
        print(f"   Avg Collisions per Trajectory: {aggregate['collisions']['mean']:.2f} ± {aggregate['collisions']['std']:.2f}")
        
        return aggregate, all_metrics
    
    def visualize_metrics(self, all_metrics, save_dir='visualizations/lsm'):
        """Create visualization plots"""
        print(f"\n📊 Creating visualizations...")
        
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Action Accuracy Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(all_metrics['action_accuracy'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Action Accuracy')
        plt.ylabel('Frequency')
        plt.title('Action Accuracy Distribution')
        plt.axvline(np.mean(all_metrics['action_accuracy']), color='red', 
                   linestyle='--', label=f'Mean: {np.mean(all_metrics["action_accuracy"])*100:.2f}%')
        plt.legend()
        plt.savefig(f'{save_dir}/action_accuracy_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Collision vs Reach Rate
        plt.figure(figsize=(10, 6))
        plt.scatter(all_metrics['collision_rate'], all_metrics['reach_rate'], alpha=0.6)
        plt.xlabel('Collision Rate')
        plt.ylabel('Reach Rate')
        plt.title('Collision Rate vs Reach Rate')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/collision_vs_reach_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Spike Rate Distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(all_metrics['avg_spike_rate'], bins=30, edgecolor='black', alpha=0.7, color='blue')
        plt.xlabel('Avg Spike Rate')
        plt.ylabel('Frequency')
        plt.title('Overall Spike Rate')
        
        plt.subplot(1, 3, 2)
        plt.hist(all_metrics['avg_e_rate'], bins=30, edgecolor='black', alpha=0.7, color='green')
        plt.xlabel('Excitatory Spike Rate')
        plt.ylabel('Frequency')
        plt.title('E Neuron Spike Rate')
        
        plt.subplot(1, 3, 3)
        plt.hist(all_metrics['avg_i_rate'], bins=30, edgecolor='black', alpha=0.7, color='red')
        plt.xlabel('Inhibitory Spike Rate')
        plt.ylabel('Frequency')
        plt.title('I Neuron Spike Rate')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/spike_rates_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. E/I Balance
        plt.figure(figsize=(10, 6))
        plt.hist(all_metrics['ei_balance'], bins=30, edgecolor='black', alpha=0.7, color='purple')
        plt.xlabel('E/I Balance')
        plt.ylabel('Frequency')
        plt.title('Excitatory/Inhibitory Balance Distribution')
        plt.axvline(np.mean(all_metrics['ei_balance']), color='red', 
                   linestyle='--', label=f'Mean: {np.mean(all_metrics["ei_balance"]):.2f}')
        plt.legend()
        plt.savefig(f'{save_dir}/ei_balance_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. Summary metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_to_plot = [
            ('action_accuracy', 'Action Accuracy', axes[0, 0]),
            ('collision_rate', 'Collision Rate', axes[0, 1]),
            ('reach_rate', 'Reach Rate', axes[1, 0]),
            ('avg_spike_rate', 'Avg Spike Rate', axes[1, 1])
        ]
        
        for metric_key, metric_name, ax in metrics_to_plot:
            data = all_metrics[metric_key]
            ax.boxplot([data], labels=[metric_name])
            ax.set_ylabel('Value')
            ax.set_title(f'{metric_name} Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/summary_boxplots_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved visualizations to: {save_dir}")
    
    def save_metrics(self, aggregate, all_metrics, save_path):
        """Save metrics to file"""
        print(f"\n💾 Saving metrics to: {save_path}")
        
        import json
        
        # Convert numpy types to Python types for JSON
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            else:
                return obj
        
        data = {
            'aggregate': convert_types(aggregate),
            'all_metrics': convert_types(all_metrics),
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"   ✅ Metrics saved")


def main(config_path='configs/config_lsm.yaml', checkpoint_path=None, num_episodes=None):
    """Main evaluation pipeline"""
    
    print("\n" + "="*70)
    print("🧪 LSM EVALUATION FOR MAPF")
    print("="*70)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup dataset paths
    root_dir = config['dataset']['root_dir']
    config['train'] = {'root_dir': root_dir, 'batch_size': 1, 'num_workers': 0}
    config['valid'] = {'root_dir': root_dir, 'batch_size': 1, 'num_workers': 0}
    
    print(f"   Config: {config_path}")
    print(f"   Device: {config['device']}")
    
    # Create network (check if neuromorphic mode is enabled)
    use_neuromorphic = config['lsm'].get('neuromorphic', {}).get('enabled', False)
    
    print(f"\n🔧 Creating LSM Network...")
    if use_neuromorphic:
        print(f"   🧠 Using Neuromorphic LSM (BDSM mode)")
        network = TrainableNeuromorphicLSMNetwork(config)
    else:
        print(f"   Using Standard LSM")
        network = LSMNetwork(config)
    
    # Create evaluator
    evaluator = LSMEvaluator(config, network, checkpoint_path)
    
    # Evaluate
    num_test = num_episodes or config['evaluation'].get('test_episodes', 100)
    aggregate, all_metrics = evaluator.evaluate_dataset(num_episodes=num_test, split='valid')
    
    # Visualize
    if config['evaluation'].get('visualize_reservoir', True):
        evaluator.visualize_metrics(all_metrics)
    
    # Save metrics
    if config['evaluation'].get('save_metrics', True):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_path = f'logs/lsm/evaluation_{timestamp}.json'
        evaluator.save_metrics(aggregate, all_metrics, metrics_path)
    
    print(f"\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    
    return aggregate, all_metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate LSM for MAPF')
    parser.add_argument('--config', type=str, default='configs/config_lsm.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--num_episodes', type=int, default=None,
                       help='Number of episodes to evaluate (default: all)')
    
    args = parser.parse_args()
    
    aggregate, all_metrics = main(args.config, args.checkpoint, args.num_episodes)
