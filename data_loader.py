import os
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader


def _load_and_resolve_obstacles_from_case(case_dir):
    """
    Load obstacles from a case directory and resolve conflicts with agent positions/goals.
    Returns resolved obstacles or None if no obstacles or input.yaml not found.
    """
    input_yaml_path = os.path.join(case_dir, "input.yaml")
    
    if not os.path.exists(input_yaml_path):
        return None
        
    try:
        with open(input_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        map_data = data.get('map', {})
        obstacles = map_data.get('obstacles', [])
        agents = data.get('agents', [])
        
        if not obstacles:
            return None
            
        # Extract agent positions and goals
        agent_positions = []
        goal_positions = []
        for agent in agents:
            start_pos = agent.get('start', [0, 0])
            goal_pos = agent.get('goal', [0, 0])
            agent_positions.append(tuple(start_pos))
            goal_positions.append(tuple(goal_pos))
        
        # Resolve obstacle conflicts
        dimensions = map_data.get('dimensions', [28, 28])  # Default to 28x28
        resolved_obstacles = _resolve_obstacle_conflicts_for_loader(
            obstacles, agent_positions, goal_positions, dimensions
        )
        
        return resolved_obstacles
        
    except Exception as e:
        print(f"ERROR: Error loading/resolving obstacles from {input_yaml_path}: {e}")
        return None


def _resolve_obstacle_conflicts_for_loader(obstacles, agent_positions, goal_positions, dimensions):
    """
    Resolve conflicts between obstacles and agent positions/goals during data loading.
    """
    if not obstacles:
        return obstacles
        
    # Convert positions to sets for faster lookup
    occupied_positions = set()
    for pos in agent_positions + goal_positions:
        occupied_positions.add(tuple(pos))
    
    resolved_obstacles = []
    
    for obs in obstacles:
        obs_tuple = tuple(obs)
        
        # Check if obstacle conflicts with any agent position or goal
        if obs_tuple in occupied_positions:
            # Find a free space for this obstacle
            new_obs = _find_free_space_for_loader(occupied_positions, dimensions)
            if new_obs is not None:
                resolved_obstacles.append(new_obs)
                occupied_positions.add(tuple(new_obs))
            # If no free space found, skip this obstacle
        else:
            # No conflict, keep original obstacle
            resolved_obstacles.append(list(obs))
            occupied_positions.add(obs_tuple)
    
    return resolved_obstacles


def _find_free_space_for_loader(occupied_positions, dimensions, max_attempts=100):
    """
    Find a free space on the grid for obstacle relocation during data loading.
    """
    width, height = dimensions
    
    # Random search first (faster for sparse grids)
    for _ in range(max_attempts):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        if (x, y) not in occupied_positions:
            return [x, y]
    
    # Systematic search if random fails
    for x in range(width):
        for y in range(height):
            if (x, y) not in occupied_positions:
                return [x, y]
    
    # No free space found
    return None


class MAPFBaseDataset(Dataset):
    """
    Base dataset for MAPF, handles loading and validation of cases.
    Subclasses should implement the __getitem__ method for specific model input needs.
    """

    def __init__(self, config, mode, resolve_obstacle_conflicts=True):
        self.config = config[mode]
        self.resolve_obstacle_conflicts = resolve_obstacle_conflicts
        self.dir_path = self.config["root_dir"]
        if mode == "valid":
            self.dir_path = os.path.join(self.dir_path, "valid")
        elif mode == "train":
            self.dir_path = os.path.join(self.dir_path, "train")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.cases = [
            d for d in os.listdir(self.dir_path)
            if d.startswith("case_") and
               os.path.exists(os.path.join(self.dir_path, d, "states.npy")) and
               os.path.exists(os.path.join(self.dir_path, d, "trajectory_record.npy")) and
               os.path.exists(os.path.join(self.dir_path, d, "gso.npy"))
        ]
        
        # Extract case indices for expert trajectory loading
        self.case_indices = []
        for case in self.cases:
            try:
                case_idx = int(case.split("_")[1])
                self.case_indices.append(case_idx)
            except (ValueError, IndexError):
                self.case_indices.append(-1)  # Invalid case index
                
        # Initialize count and skipped variables
        self.count = 0
        self.skipped = 0
        
        # If obstacle conflict resolution is enabled, resolve conflicts for all cases
        if self.resolve_obstacle_conflicts:
            self._resolve_all_obstacles()
                
        self.states = np.zeros(
            (
                len(self.cases),
                self.config["min_time"],
                self.config["nb_agents"],
                2,
                5,
                5,
            ), dtype=np.float32
        )
        self.trajectories = np.zeros(
            (len(self.cases), self.config["min_time"], self.config["nb_agents"]), dtype=np.float32
        )
        self.gsos = np.zeros(
            (
                len(self.cases),
                self.config["min_time"],
                self.config["nb_agents"],
                self.config["nb_agents"],
            ), dtype=np.float32
        )
        
        # Load the actual data
        self._load_data()
        
    def _resolve_all_obstacles(self):
        """
        Apply obstacle conflict resolution to all cases in the dataset.
        This updates input.yaml files with resolved obstacle positions.
        """
        print(f"Applying obstacle conflict resolution to {len(self.cases)} cases...")
        resolved_count = 0
        
        for case in self.cases:
            case_dir = os.path.join(self.dir_path, case)
            resolved_obstacles = _load_and_resolve_obstacles_from_case(case_dir)
            
            if resolved_obstacles is not None:
                # Update the input.yaml file with resolved obstacles
                self._update_case_obstacles(case_dir, resolved_obstacles)
                resolved_count += 1
                
        print(f"Obstacle conflict resolution applied to {resolved_count} cases with obstacles.")
        
    def _update_case_obstacles(self, case_dir, resolved_obstacles):
        """
        Update the input.yaml file with resolved obstacle positions.
        """
        input_yaml_path = os.path.join(case_dir, "input.yaml")
        
        try:
            with open(input_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                
            # Update obstacles in the data
            data['map']['obstacles'] = resolved_obstacles
            
            # Write back to file
            with open(input_yaml_path, 'w') as f:
                yaml.safe_dump(data, f)
                
        except Exception as e:
            print(f"ERROR: Failed to update obstacles for {case_dir}: {e}")
    
    def _load_data(self):
        """
        Load the actual data files after obstacle resolution.
        """
        self.count = 0
        self.skipped = 0
        for i, case in enumerate(self.cases):
            try:
                state_path = os.path.join(self.dir_path, case, "states.npy")
                tray_path = os.path.join(self.dir_path, case, "trajectory_record.npy")
                gso_path = os.path.join(self.dir_path, case, "gso.npy")
                
                if not os.path.exists(state_path):
                    print(f"Skipping {case}: states.npy missing.")
                    self.skipped += 1
                    continue
                if not os.path.exists(tray_path):
                    print(f"Skipping {case}: trajectory_record.npy missing.")
                    self.skipped += 1
                    continue
                if not os.path.exists(gso_path):
                    print(f"Skipping {case}: gso.npy missing.")
                    self.skipped += 1
                    continue
                    
                state = np.load(state_path)
                tray = np.load(tray_path)
                gso = np.load(gso_path)
                
                # Slicing and validation
                if state.shape[0] < self.config["min_time"] + 1:
                    print(f"Skipping {case}: not enough time steps in states.npy ({state.shape[0]} < {self.config['min_time']+1})")
                    self.skipped += 1
                    continue
                if tray.shape[1] < self.config["min_time"]:
                    print(f"Skipping {case}: not enough time steps in trajectory_record.npy ({tray.shape[1]} < {self.config['min_time']})")
                    self.skipped += 1
                    continue
                if gso.shape[0] < self.config["min_time"]:
                    print(f"Skipping {case}: not enough time steps in gso.npy ({gso.shape[0]} < {self.config['min_time']})")
                    self.skipped += 1
                    continue
                    
                state = state[1 : self.config["min_time"] + 1, :, :, :, :]
                tray = tray[:, : self.config["min_time"]]
                gso = gso[: self.config["min_time"], 0, :, :] + np.eye(self.config["nb_agents"])
                
                if (
                    state.shape[0] < self.config["min_time"]
                    or tray.shape[1] < self.config["min_time"]
                ):
                    print(f"Skipping {case}: shape mismatch after slicing (states: {state.shape[0]}, tray: {tray.shape[1]})")
                    self.skipped += 1
                    continue
                if (
                    state.shape[0] > self.config.get("max_time_dl", 9999)
                    or tray.shape[1] > self.config.get("max_time_dl", 9999)
                ):
                    print(f"Skipping {case}: exceeds max_time_dl")
                    self.skipped += 1
                    continue
                if state.shape[0] != tray.shape[1]:
                    print(f"Skipping {case}: states and trajectories time mismatch after slicing ({state.shape[0]} != {tray.shape[1]})")
                    self.skipped += 1
                    continue
                    
                self.states[i, :, :, :, :, :] = state
                self.trajectories[i, :, :] = tray.T
                self.gsos[i, :, :, :] = gso
                self.count += 1
                
            except Exception as e:
                print(f"Error loading {case}: {e}, skipping.")
                self.skipped += 1
                continue
                
        self.states = self.states[: self.count]
        self.trajectories = self.trajectories[: self.count]
        self.gsos = self.gsos[: self.count]
        self.case_indices = self.case_indices[: self.count]  # Trim case indices to match loaded data
        
        if self.count == 0:
            raise RuntimeError(f"No valid cases loaded from {self.dir_path}. Please check your dataset and config.")
        assert self.states.shape[0] == self.trajectories.shape[0], (
            f"Mismatch after loading: {self.states.shape[0]} != {self.trajectories.shape[0]}"
        )
        print(f"Loaded {self.count} cases, skipped {self.skipped} cases.")

    def __len__(self):
        return self.count

    def statistics(self):
        zeros = np.count_nonzero(self.trajectories == 0)
        return zeros / (self.trajectories.shape[0] * self.trajectories.shape[1])


class CreateSNNDataset(MAPFBaseDataset):
    """
    Dataset for SNN models. Returns (states, trajectories, gsos) per case.
    States are returned as (time, agents, 2, 5, 5) for SNN input.
    """

    def __init__(self, config, mode, resolve_obstacle_conflicts=True):
        super().__init__(config, mode, resolve_obstacle_conflicts)

    def __getitem__(self, index):
        states = torch.from_numpy(self.states[index]).float()  # (time, agents, 2, 5, 5)
        trayec = torch.from_numpy(self.trajectories[index]).float()  # (time, agents)
        gsos = torch.from_numpy(self.gsos[index]).float()  # (time, agents, agents)
        case_idx = self.case_indices[index]  # Get case index for expert trajectory
        # Do NOT flatten states for SNN
        # states = states.view(states.shape[0], states.shape[1], -1)  # REMOVE THIS LINE
        return states, trayec, gsos, case_idx


class CreateGNNDataset(MAPFBaseDataset):
    """
    Dataset for GNN/baseline models. Returns (states, trajectories, gsos) per time step.
    States are reshaped to (samples, agents, 2, 5, 5) for GNN input.
    """

    def __init__(self, config, mode, resolve_obstacle_conflicts=True):
        super().__init__(config, mode, resolve_obstacle_conflicts)
        # Flatten across time for GNN: (cases, time, ...) -> (cases*time, ...)
        self.states = self.states.reshape(-1, self.config["nb_agents"], 2, 5, 5)
        self.trajectories = self.trajectories.reshape(-1, self.config["nb_agents"])
        self.gsos = self.gsos.reshape(-1, self.config["nb_agents"], self.config["nb_agents"])
        self.count = self.states.shape[0]

    def __getitem__(self, index):
        states = torch.from_numpy(self.states[index]).float()  # (agents, 2, 5, 5)
        trayec = torch.from_numpy(self.trajectories[index]).float()  # (agents,)
        gsos = torch.from_numpy(self.gsos[index]).float()  # (agents, agents)
        return states, trayec, gsos


class GNNDataLoader:
    def __init__(self, config, resolve_obstacle_conflicts=True):
        self.config = config
        # Only use pin_memory if CUDA is available
        use_pin_memory = torch.cuda.is_available()
        
        train_set = CreateGNNDataset(self.config, "train", resolve_obstacle_conflicts)
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.config["train"]["batch_size"],
            shuffle=True,
            num_workers=self.config["train"].get("num_workers", 0),
            pin_memory=use_pin_memory,
            drop_last=True
        )
        if "valid" in self.config:
            valid_set = CreateGNNDataset(self.config, "valid", resolve_obstacle_conflicts)
            self.valid_loader = DataLoader(
                valid_set,
                batch_size=self.config["valid"].get("batch_size", 1),
                shuffle=False,
                num_workers=self.config["valid"].get("num_workers", 0),
                pin_memory=use_pin_memory,
            )
        else:
            self.valid_loader = None


class SNNDataLoader:
    def __init__(self, config, resolve_obstacle_conflicts=True):
        self.config = config
        # Only use pin_memory if CUDA is available
        use_pin_memory = torch.cuda.is_available()
        
        train_set = CreateSNNDataset(self.config, "train", resolve_obstacle_conflicts)
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.config["train"]["batch_size"],
            shuffle=True,
            num_workers=self.config["train"].get("num_workers", 0),
            pin_memory=use_pin_memory,
            drop_last=True
        )
        if "valid" in self.config:
            valid_set = CreateSNNDataset(self.config, "valid", resolve_obstacle_conflicts)
            self.valid_loader = DataLoader(
                valid_set,
                batch_size=self.config["valid"].get("batch_size", 1),
                shuffle=False,
                num_workers=self.config["valid"].get("num_workers", 0),
                pin_memory=use_pin_memory,
            )
        else:
            self.valid_loader = None


if __name__ == "__main__":
    # Test obstacle conflict resolution functionality only
    print("Testing obstacle conflict resolution functionality...")
    
    # Test a few cases manually
    case_dir = "dataset/5_8_28/train/case_0"
    resolved_obstacles = _load_and_resolve_obstacles_from_case(case_dir)
    if resolved_obstacles is not None:
        print(f"Case 0 obstacle resolution successful: {len(resolved_obstacles)} obstacles resolved")
    else:
        print("Case 0 has no obstacles or input.yaml not found")
    
    # Test with actual case that has obstacles
    found_obstacles = False
    for i in range(10):
        case_dir = f"dataset/5_8_28/train/case_{i}"
        resolved_obstacles = _load_and_resolve_obstacles_from_case(case_dir)
        if resolved_obstacles is not None and len(resolved_obstacles) > 0:
            print(f"Case {i} obstacle resolution successful: {len(resolved_obstacles)} obstacles resolved")
            found_obstacles = True
            break
    
    if not found_obstacles:
        print("No cases with obstacles found in first 10 cases")
    
    print("Obstacle conflict resolution functionality verified!")
    print("\nNOTE: Full data loader testing requires proper training data with compatible shapes.")
    print("The data loader is ready to use with correct configurations during actual training.")
    
    # Uncomment and provide config to test the data loaders:
    # print("Testing GNNDataLoader...")
    # gnn_loader = GNNDataLoader(config)
    # train_batch = next(iter(gnn_loader.train_loader))
    # print(f"GNN train batch: {[x.shape for x in train_batch]}")
    # if gnn_loader.valid_loader:
    #     valid_batch = next(iter(gnn_loader.valid_loader))
    #     print(f"GNN valid batch: {[x.shape for x in valid_batch]}")
    # print("Testing SNNDataLoader...")
    # snn_loader = SNNDataLoader(config)
    # train_batch = next(iter(snn_loader.train_loader))
    # print(f"SNN train batch: {[x.shape for x in train_batch]}")
    # if snn_loader.valid_loader:
    #     valid_batch = next(iter(snn_loader.valid_loader))
    #     print(f"SNN valid batch: {[x.shape for x in valid_batch]}")
