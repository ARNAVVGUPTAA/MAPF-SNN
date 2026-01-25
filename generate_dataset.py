#!/usr/bin/env python3
"""
Generate larger MAPF dataset
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.dataset_gen import create_solutions
from data_generation.trayectory_parser import parse_traject
from data_generation.record import record_env
from config import config
import torch

if __name__ == "__main__":
    print("🚀 Generating Large MAPF Dataset")
    print("=" * 60)
    
    # Set device
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Generate LARGE dataset with 80-20 split
    cases_train = 8000   # 8k training cases (80%)
    cases_val = 2000     # 2k validation cases (20%)
    
    print(f"📊 Training cases: {cases_train}")
    print(f"📊 Validation cases: {cases_val}")
    print("=" * 60)
    
    # Training config
    config_train = {
        "path": "dataset/5_8_28/train",
        "nb_agents": 5,
        "num_agents": 5,  # Required by GraphEnv
        "nb_obstacles": 0,
        "sensor_range": 5,
        "board_size": [9, 9],
        "map_shape": (9, 9),
        "min_time": 4,
        "max_time": 60
    }
    
    # Validation config
    config_val = {
        "path": "dataset/5_8_28/valid",
        "nb_agents": 5,
        "num_agents": 5,  # Required by GraphEnv
        "nb_obstacles": 0,
        "sensor_range": 5,
        "board_size": [9, 9],
        "map_shape": (9, 9),
        "min_time": 4,
        "max_time": 60
    }
    
    # Generate training data
    print("\n🔧 Generating training data...")
    create_solutions(config_train["path"], cases_train, config_train)
    print("📝 Parsing trajectories...")
    parse_traject(config_train["path"])
    print("💾 Recording environment data...")
    record_env(config_train["path"], config_train)
    
    # Generate validation data
    print("\n🔧 Generating validation data...")
    create_solutions(config_val["path"], cases_val, config_val)
    print("📝 Parsing trajectories...")
    parse_traject(config_val["path"])
    print("💾 Recording environment data...")
    record_env(config_val["path"], config_val)
    
    print("\n✅ Dataset generation complete!")
    print(f"📁 Training: dataset/5_8_28/train ({cases_train} cases)")
    print(f"📁 Validation: dataset/5_8_28/valid ({cases_val} cases)")
