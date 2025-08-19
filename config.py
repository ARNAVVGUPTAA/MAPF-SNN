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
    config = load_config('clean_release/configs/config_snn.yaml')
except FileNotFoundError:
    config = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
