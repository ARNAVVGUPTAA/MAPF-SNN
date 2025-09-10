import argparse
import yaml
import torch
import os
from pathlib import Path

def load_config(config_path=None):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str, optional): Path to the YAML configuration file.
                                   If None, uses default from command line args.
    
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        parser = argparse.ArgumentParser(description="Global configuration")
        parser.add_argument(
            '--config',
            type=str,
            default="configs/config_snn.yaml",
            help="Path to the YAML configuration file"
        )
        args, _ = parser.parse_known_args()
        config_path = args.config
    
    # Make path relative to this script's directory
    script_dir = Path(__file__).parent
    if not os.path.isabs(config_path):
        config_path = script_dir / config_path
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Global device assignment
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    return config

# Global configuration loader for backward compatibility
try:
    parser = argparse.ArgumentParser(description="Global configuration")
    parser.add_argument(
        '--config',
        type=str,
        default="configs/config_snn.yaml",
        help="Path to the YAML configuration file"
    )
    args, _ = parser.parse_known_args()

    # Try to load config, with fallback to create default
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"✅ Config loaded from: {config_path}")
    else:
        # Create minimal default config
        print(f"⚠️  Config file not found at {config_path}, using defaults")
        config = {
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            'batch_size': 8,
            'epochs': 100,
            'sequence_length': 100,
            'board_size': [9, 9],
            'use_amp': True,
            'gradient_clip_norm': 0.1,
            'gradient_clip_value': 0.05,
            'use_gradient_scaling': True,
            'loss_scale_factor': 0.01,
            'use_nan_protection': True,
            'goal_reward': 10.0,
            'collision_punishment': -8.0,
            'step_penalty': -0.5,
            'cooperation_bonus': 3.0,
            'reward_decay': 0.95,
            'trace_length': 10,
            'reward_scale': 0.2,
            'punishment_scale': 0.5
        }
        
except Exception as e:
    print(f"⚠️  Error loading config: {e}, using minimal defaults")
    config = {
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'batch_size': 8,
        'epochs': 100,
        'sequence_length': 100,
        'board_size': [9, 9],
        'use_amp': True
    }

# Make config values easily accessible
# e.g., config['lif_tau'], config['adapt_alpha'], etc.
