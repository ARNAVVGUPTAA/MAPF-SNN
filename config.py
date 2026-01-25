import yaml, torch

def load_config(path=None):
    cfg_path = path
    if cfg_path is None:
        raise ValueError('Path must be provided to load_config in clean release')
    with open(cfg_path,'r') as f: cfg=yaml.safe_load(f)
    cfg['device']= 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg

# Provide a module-level default config if desired; not auto-parsing CLI here to keep minimal.
try:
    config = load_config('configs/config_snn.yaml')
except FileNotFoundError:
    # Fallback basic config for clean operation
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 50,
        'batch_size': 8,
        'learning_rate': 1e-4,  # Reduced by 10x from 1e-3
        'weight_decay': 1e-5,
        'sequence_length': 100,
        'board_size': [9, 9],
        'num_actions': 5,
        'num_agents': 5,
        'hidden_dim': 128,
        'input_dim': 147,
        'reward_scale': 0.001,       # ULTRA reduced - avg reward was still 26-30!
        'punishment_scale': 0.15,    # Increase punishment scale for better balance
        'goal_reward': 100.0,         # HUGE reward boost for reaching goal
        'collision_punishment': -3.0,  # 10x INCREASED from -0.3 (strong collision penalty)
        'step_penalty': -1.0,        # 10x INCREASED from -0.1 (strong step penalty)
        'cooperation_bonus': 1.0,
        
        # Spike rate penalty parameters - NEW!
        'spike_rate_penalty': -2.0,  # Penalty when spike rate falls below threshold
        'min_spike_rate_threshold': 0.03,  # 3% minimum spike rate threshold
        
        # High spike activity penalties - NEW!
        'high_spike_penalty': -1.5,  # REDUCED from -3.0 - Gentler penalty for excessive spiking
        'max_spike_rate_threshold': 0.15,  # 15% maximum spike rate threshold
        'reward_reduction_factor': 0.7,  # INCREASED from 0.5 - Less aggressive reward reduction
        
        'reward_decay': 0.95,
        'trace_length': 13,
        
        # Entropy regularization for exploration
        'entropy_coefficient': 0.5,  # 100x INCREASED from 0.01 - Strong exploration bonus
        
        # Activity regularization to prevent neuron death
        'activity_regularization_strength': 0.01,  # REDUCED from 0.05 - Looser activity regularization
        
        # Input noise for early training epochs
        'input_noise_level': 0.01,  # Gaussian noise added to FOV inputs during first 5 epochs
        
        # Reward scaling for gradient stability (HIGHEST PRIORITY FIX)
        'reward_scale_factor': 20.0,  # Divide raw rewards by this to prevent gradient explosion
    }
