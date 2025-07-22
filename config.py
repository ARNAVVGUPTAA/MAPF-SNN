import argparse
import yaml
import torch

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
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Global device assignment
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    return config

# Global configuration loader for backward compatibility
parser = argparse.ArgumentParser(description="Global configuration")
parser.add_argument(
    '--config',
    type=str,
    default="configs/config_snn.yaml",
    help="Path to the YAML configuration file"
)
args, _ = parser.parse_known_args()

config = load_config(args.config)

# Make config values easily accessible
# e.g., config['lif_tau'], config['adapt_alpha'], etc.
