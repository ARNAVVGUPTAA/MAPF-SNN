"""
LSM Training Script for MAPF
==============================

Reservoir Computing Approach:
1. Load expert trajectories from dataset
2. Pass observations through FIXED reservoir
3. Collect liquid states (spike accumulations)
4. Train LINEAR readout using ridge regression
5. Evaluate on test set

The reservoir itself is NOT trained - only the readout layer.
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lsm_reservoir import LSMNetwork, CircularMeshReservoir
from models.neuromorphic_lsm_trainable import TrainableNeuromorphicLSMNetwork
from data_loader import SNNDataset, pad_collate_fn


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add device
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup paths for train/valid datasets
    root_dir = config['dataset']['root_dir']
    config['train'] = {
        'root_dir': root_dir,
        'batch_size': 1,  # Process one trajectory at a time
        'num_workers': 0
    }
    config['valid'] = {
        'root_dir': root_dir,
        'batch_size': 1,
        'num_workers': 0
    }
    
    return config


class LiquidStateCollector:
    """
    Collects liquid states from expert trajectories.
    
    Flow:
    1. Load trajectory from dataset (states, actions)
    2. Pass each observation through reservoir
    3. Record liquid state + expert action
    4. Build dataset: X (liquid states) → Y (expert actions)
    """
    
    def __init__(self, config, reservoir_network):
        self.config = config
        self.reservoir_network = reservoir_network
        self.device = config['device']
        
        # Move network to device
        self.reservoir_network.to(self.device)
        self.reservoir_network.eval()  # No training during collection
        
        # Storage
        self.X_states = []  # Liquid states
        self.Y_actions = []  # Expert actions
        
    def collect_from_dataset(self, max_episodes=None):
        """
        Collect liquid states from dataset trajectories.
        
        Args:
            max_episodes: Maximum number of episodes to collect (None = all)
        """
        print("\n" + "="*70)
        print("🔬 COLLECTING LIQUID STATES FROM EXPERT TRAJECTORIES")
        print("="*70)
        
        # Load dataset
        dataset = SNNDataset(self.config, 'train')
        
        if max_episodes is None:
            max_episodes = len(dataset)
        else:
            max_episodes = min(max_episodes, len(dataset))
        
        print(f"   Processing {max_episodes} trajectories...")
        
        total_timesteps = 0
        successful_episodes = 0
        
        with torch.no_grad():
            for ep_idx in tqdm(range(max_episodes), desc="Collecting"):
                # Get trajectory
                states, actions, gso = dataset[ep_idx]
                
                # states: [T, A, 2, H, W]
                # actions: [T, A]
                T, A = actions.shape[:2]
                
                # Reset reservoir state for new trajectory
                self.reservoir_network.reset_state()
                
                # Process each timestep
                for t in range(T):
                    # Get observations for all agents at this timestep
                    fov = states[t]  # [A, 2, H, W]
                    fov = fov.to(self.device)
                    
                    # Process each agent separately (they have independent FOVs)
                    for agent_id in range(A):
                        agent_fov = fov[agent_id:agent_id+1]  # [1, 2, H, W]
                        
                        # Get liquid state
                        liquid_state = self.reservoir_network.get_liquid_state(agent_fov)
                        liquid_state = liquid_state.cpu().numpy()  # [1, neurons]
                        
                        # Get expert action
                        expert_action = actions[t, agent_id].item()
                        
                        # Store
                        self.X_states.append(liquid_state[0])  # [neurons]
                        self.Y_actions.append(expert_action)
                        
                        total_timesteps += 1
                
                successful_episodes += 1
                
                # Progress update
                if (ep_idx + 1) % 100 == 0:
                    print(f"   Collected {successful_episodes} episodes, {total_timesteps} timesteps")
        
        print(f"\n✅ Collection Complete!")
        print(f"   Episodes: {successful_episodes}")
        print(f"   Total timesteps: {total_timesteps}")
        print(f"   Liquid states shape: ({len(self.X_states)}, {len(self.X_states[0])})")
        
        return np.array(self.X_states), np.array(self.Y_actions)


def train_readout_ridge_regression(X, Y, lambda_reg=1e-2):
    """
    Train linear readout using ridge regression.
    
    Ridge Regression: W = (X^T X + λI)^{-1} X^T Y
    
    Args:
        X: [N, num_neurons] liquid states
        Y: [N] expert actions (integer labels)
        lambda_reg: Ridge regularization parameter
    
    Returns:
        W: [num_neurons+1, num_actions] readout weights (includes bias)
    """
    print("\n" + "="*70)
    print("🎓 TRAINING READOUT WITH RIDGE REGRESSION")
    print("="*70)
    
    N = X.shape[0]
    num_neurons = X.shape[1]
    num_actions = int(Y.max()) + 1
    
    print(f"   Dataset: {N} samples")
    print(f"   Features: {num_neurons} neurons")
    print(f"   Actions: {num_actions} classes")
    print(f"   Regularization: λ = {lambda_reg}")
    
    # Convert action labels to one-hot
    Y_onehot = np.zeros((N, num_actions))
    Y_onehot[np.arange(N), Y.astype(int)] = 1
    
    # Add bias term to X
    X_bias = np.hstack([X, np.ones((N, 1))])
    
    print(f"\n   Computing (X^T X + λI)^{{-1}}...")
    # Ridge regression solution
    XtX = X_bias.T @ X_bias
    I = np.eye(X_bias.shape[1])
    XtX_reg = XtX + lambda_reg * I
    
    print(f"   Computing X^T Y...")
    XtY = X_bias.T @ Y_onehot
    
    print(f"   Solving linear system...")
    W = np.linalg.solve(XtX_reg, XtY)
    
    # Compute training accuracy
    Y_pred = X_bias @ W
    Y_pred_labels = np.argmax(Y_pred, axis=1)
    train_accuracy = np.mean(Y_pred_labels == Y)
    
    print(f"\n✅ Training Complete!")
    print(f"   Readout weights: {W.shape}")
    print(f"   Training accuracy: {train_accuracy*100:.2f}%")
    
    return W


def train_readout_pytorch(X, Y, lambda_reg=1e-2, num_epochs=50, batch_size=256, lr=1e-3):
    """
    Alternative: Train readout using PyTorch with SGD (for large datasets)
    
    Args:
        X: [N, num_neurons] liquid states
        Y: [N] expert actions
        lambda_reg: L2 regularization
        num_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        W: Readout weights
    """
    print("\n" + "="*70)
    print("🎓 TRAINING READOUT WITH PYTORCH SGD")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    N = X.shape[0]
    num_neurons = X.shape[1]
    num_actions = int(Y.max()) + 1
    
    print(f"   Dataset: {N} samples")
    print(f"   Features: {num_neurons} neurons")
    print(f"   Actions: {num_actions} classes")
    print(f"   Device: {device}")
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X).float().to(device)
    Y_tensor = torch.from_numpy(Y).long().to(device)
    
    # Create dataloader
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create linear model
    model = nn.Linear(num_neurons, num_actions).to(device)
    nn.init.normal_(model.weight, mean=0, std=0.01)
    nn.init.zeros_(model.bias)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_reg)
    
    # Training loop
    print(f"\n   Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_Y in loader:
            optimizer.zero_grad()
            
            logits = model(batch_X)
            loss = criterion(logits, batch_Y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch_Y).sum().item()
            total += batch_Y.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={accuracy*100:.2f}%")
    
    # Extract weights
    W = model.weight.data.cpu().numpy().T  # [neurons, actions]
    b = model.bias.data.cpu().numpy().reshape(1, -1)  # [1, actions]
    W_full = np.vstack([W, b])  # [neurons+1, actions]
    
    print(f"\n✅ Training Complete!")
    print(f"   Final training accuracy: {accuracy*100:.2f}%")
    
    return W_full


def evaluate_readout(X, Y, W, split_name="Test"):
    """
    Evaluate readout performance.
    
    Args:
        X: [N, num_neurons] liquid states
        Y: [N] true actions
        W: [num_neurons+1, num_actions] readout weights
        split_name: Name of split for printing
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print(f"\n📊 Evaluating on {split_name} Set...")
    
    N = X.shape[0]
    
    # Add bias
    X_bias = np.hstack([X, np.ones((N, 1))])
    
    # Predict
    Y_pred = X_bias @ W
    Y_pred_labels = np.argmax(Y_pred, axis=1)
    
    # Accuracy
    accuracy = np.mean(Y_pred_labels == Y)
    
    # Per-class accuracy
    num_actions = W.shape[1]
    per_class_acc = []
    for action in range(num_actions):
        mask = Y == action
        if mask.sum() > 0:
            acc = np.mean(Y_pred_labels[mask] == Y[mask])
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0.0)
    
    metrics = {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'num_samples': N
    }
    
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Per-class: {[f'{a*100:.1f}%' for a in per_class_acc]}")
    
    return metrics


def main(config_path='configs/config_lsm.yaml'):
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("🚀 LSM TRAINING PIPELINE FOR MAPF")
    print("="*70)
    
    # Load config
    config = load_config(config_path)
    print(f"\n✅ Config loaded from: {config_path}")
    print(f"   Device: {config['device']}")
    
    # Check if neuromorphic mode is enabled
    use_neuromorphic = config['lsm'].get('neuromorphic', {}).get('enabled', False)
    training_mode = config['training']['mode']
    
    # Create reservoir network
    print(f"\n🔧 Creating LSM Network...")
    if use_neuromorphic:
        print(f"   🧠 Using Neuromorphic LSM (BDSM: Brain Don't Simply Multiply)")
        print(f"   Mode: {training_mode}")
        network = TrainableNeuromorphicLSMNetwork(config)
    else:
        print(f"   Using Standard LSM")
        network = LSMNetwork(config)
    
    # Freeze reservoir (only readout is trainable) for reservoir computing mode
    if training_mode == 'reservoir_computing':
        for name, param in network.named_parameters():
            if 'readout' not in name:
                param.requires_grad = False
        print(f"   Reservoir parameters: FROZEN")
        print(f"   Readout parameters: TRAINABLE")
    else:
        print(f"   All parameters: TRAINABLE (full SGD/BPTT mode)")
    
    # Collect liquid states
    collector = LiquidStateCollector(config, network)
    
    max_episodes = config['readout'].get('collect_episodes', None)
    X, Y = collector.collect_from_dataset(max_episodes=max_episodes)
    
    # Save collected data
    os.makedirs('data/lsm', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_path = f'data/lsm/liquid_states_{timestamp}.npz'
    np.savez(data_path, X=X, Y=Y)
    print(f"\n💾 Saved liquid states to: {data_path}")
    
    # Split into train/validation
    val_split = config['readout']['validation_split']
    N = len(X)
    N_train = int(N * (1 - val_split))
    
    # Shuffle
    indices = np.random.permutation(N)
    train_indices = indices[:N_train]
    val_indices = indices[N_train:]
    
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_val, Y_val = X[val_indices], Y[val_indices]
    
    print(f"\n📊 Dataset Split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    
    # Train readout
    method = config['readout']['method']
    lambda_reg = float(config['readout']['regularization'])
    
    if method == 'ridge_regression':
        W = train_readout_ridge_regression(X_train, Y_train, lambda_reg=lambda_reg)
    elif method == 'pytorch':
        W = train_readout_pytorch(X_train, Y_train, lambda_reg=lambda_reg)
    else:
        raise ValueError(f"Unknown readout method: {method}")
    
    # Evaluate
    print(f"\n" + "="*70)
    print("📈 EVALUATION")
    print("="*70)
    
    train_metrics = evaluate_readout(X_train, Y_train, W, "Train")
    val_metrics = evaluate_readout(X_val, Y_val, W, "Validation")
    
    # Load readout weights into network
    num_neurons = W.shape[0] - 1
    network.readout.weight.data = torch.from_numpy(W[:num_neurons].T).float()
    network.readout.bias.data = torch.from_numpy(W[num_neurons]).float()
    
    # Save model
    os.makedirs('checkpoints/lsm', exist_ok=True)
    model_path = f'checkpoints/lsm/lsm_network_{timestamp}.pt'
    torch.save({
        'config': config,
        'network_state': network.state_dict(),
        'readout_weights': W,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'timestamp': timestamp
    }, model_path)
    
    print(f"\n💾 Model saved to: {model_path}")
    
    print(f"\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"   Train Accuracy: {train_metrics['accuracy']*100:.2f}%")
    print(f"   Val Accuracy: {val_metrics['accuracy']*100:.2f}%")
    
    return network, W, train_metrics, val_metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSM for MAPF')
    parser.add_argument('--config', type=str, default='configs/config_lsm.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    network, W, train_metrics, val_metrics = main(args.config)
