import argparse
import yaml
import torch

# Global configuration loader
parser = argparse.ArgumentParser(description="Global configuration")
parser.add_argument(
    '--config',
    type=str,
    default="configs/config_snn.yaml",
    help="Path to the YAML configuration file"
)
args, _ = parser.parse_known_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Global device assignment
config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Make config values easily accessible
# e.g., config['lif_tau'], config['adapt_alpha'], etc.
