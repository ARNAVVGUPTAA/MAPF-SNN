"""
SNN Data Loader for MAPF
=========================
Loads states (FOV observations), actions (trajectory_record), and graph structure (GSO)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SNNDataset(Dataset):
    """Dataset for SNN training - loads states, actions, and GSO"""
    
    def __init__(self, cfg, mode):
        self.mode = mode
        self.cfg = cfg[mode]
        split = 'train' if mode == 'train' else 'valid'

        # Support single root_dir or list of root_dirs
        roots = self.cfg.get('root_dirs', None)
        if roots is None:
            roots = [self.cfg['root_dir']]

        # Collect (case_name, case_path) pairs from all roots
        all_case_paths = []
        for root in roots:
            data_dir = os.path.join(root, split)
            if not os.path.isdir(data_dir):
                print(f"   ⚠️  Skipping missing dir: {data_dir}")
                continue
            print(f"📂 Loading {mode} data from: {data_dir}")
            cases = sorted(d for d in os.listdir(data_dir) if d.startswith('case_'))
            print(f"   Found {len(cases)} case folders")
            all_case_paths.extend(
                (case, os.path.join(data_dir, case), root)
                for case in cases
            )

        self.data = []
        loaded_count = 0
        skipped_count = 0

        for case, case_path, root in all_case_paths:
            # Check required files
            states_file  = os.path.join(case_path, 'states.npy')
            actions_file = os.path.join(case_path, 'trajectory_record.npy')
            gso_file     = os.path.join(case_path, 'gso.npy')
            
            if not all(os.path.exists(f) for f in [states_file, actions_file, gso_file]):
                skipped_count += 1
                continue
            
            try:
                # Load the data
                states = np.load(states_file)  # [T, A, 2, H, W]
                actions = np.load(actions_file)  # [A, T] or [T, A]
                gso = np.load(gso_file)  # [T, A, A, A] or [T, A, A]
                
                # Handle different action shapes
                if actions.ndim == 2:
                    if actions.shape[0] < actions.shape[1]:
                        # [A, T] -> transpose to [T, A]
                        actions = actions.T
                
                # Handle GSO shape - take first agent's adjacency if [T, A, A, A]
                if gso.ndim == 4:
                    gso = gso[:, 0, :, :]  # [T, A, A]
                
                # Validate shapes match
                T = min(states.shape[0], actions.shape[0], gso.shape[0])
                if T == 0:
                    skipped_count += 1
                    continue
                
                # Truncate to matching length
                states = states[:T]  # [T, A, 2, H, W]
                actions = actions[:T]  # [T, A]
                gso = gso[:T]  # [T, A, A]
                
                # Add identity to GSO for self-connections
                A = gso.shape[1]
                gso = gso + np.eye(A)[np.newaxis, :, :]
                
                # Store as tuple
                self.data.append({
                    'states': states.astype(np.float32),
                    'actions': actions.astype(np.float32),
                    'gso': gso.astype(np.float32),
                    'case': case,
                    'case_dir': case_path,
                    'source_root': root,
                    'is_recovery': 'recovery' in root,
                    'timesteps': T
                })
                
                loaded_count += 1
                
                if loaded_count <= 3:  # Debug first few cases
                    print(f"   ✓ {case}: states={states.shape}, actions={actions.shape}, gso={gso.shape}, T={T}")
                
            except Exception as e:
                skipped_count += 1
                if skipped_count <= 5:  # Show first few errors
                    print(f"   ✗ {case}: Error - {e}")
                continue
        
        print(f"   ✅ Loaded {loaded_count} cases, skipped {skipped_count}")

        # Shuffle so multi-dir training mixes datasets from the start
        import random
        random.shuffle(self.data)
        
        if loaded_count == 0:
            raise RuntimeError(f'No valid data loaded for {mode}! Check dataset roots: {roots}')
        
        self.num_samples = loaded_count
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            states: [T, A, 2, H, W] - FOV observations
            actions: [T, A] - Actions taken
            gso: [T, A, A] - Graph adjacency matrices
        """
        sample = self.data[idx]
        
        states = torch.from_numpy(sample['states'])
        actions = torch.from_numpy(sample['actions'])
        gso = torch.from_numpy(sample['gso'])
        
        return states, actions, gso


def pad_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads all sequences to the maximum length in the batch.
    Also handles variable FOV sizes by padding to the largest FOV.
    """
    states_list, actions_list, gso_list = zip(*batch)
    
    # Find max timesteps in this batch
    max_T = max(s.shape[0] for s in states_list)
    batch_size = len(states_list)
    A = states_list[0].shape[1]  # Number of agents
    
    # Find max FOV size across all samples
    max_H = max(s.shape[3] for s in states_list)
    max_W = max(s.shape[4] for s in states_list)
    
    # Pre-allocate padded tensors
    padded_states = torch.zeros(batch_size, max_T, A, 2, max_H, max_W)
    padded_actions = torch.zeros(batch_size, max_T, A)
    padded_gso = torch.zeros(batch_size, max_T, A, A)
    
    # Pad each sample by repeating the last frame
    for i, (states, actions, gso) in enumerate(zip(states_list, actions_list, gso_list)):
        T = states.shape[0]
        H, W = states.shape[3], states.shape[4]
        
        # Copy actual data - pad FOV if needed
        padded_states[i, :T, :, :, :H, :W] = states
        
        # Handle actions - might have different number of agents
        actions_A = actions.shape[1]
        if actions_A == A:
            padded_actions[i, :T] = actions
        else:
            # Pad or truncate agents dimension if mismatch
            padded_actions[i, :T, :min(actions_A, A)] = actions[:, :min(actions_A, A)]
        
        # Handle gso similarly
        gso_A = gso.shape[1]
        if gso_A == A:
            padded_gso[i, :T] = gso
        else:
            padded_gso[i, :T, :min(gso_A, A), :min(gso_A, A)] = gso[:, :min(gso_A, A), :min(gso_A, A)]
        
        # Repeat last frame to pad if needed (handle FOV size mismatch)
        if T < max_T:
            # Get last frame and pad to max FOV size
            last_state = torch.zeros(1, A, 2, max_H, max_W)
            last_state[0, :, :, :H, :W] = states[-1]
            padded_states[i, T:] = last_state.repeat(max_T - T, 1, 1, 1, 1)
            
            padded_actions[i, T:] = padded_actions[i, T-1:T].repeat(max_T - T, 1)
            padded_gso[i, T:] = padded_gso[i, T-1:T].repeat(max_T - T, 1, 1)
    
    return padded_states, padded_actions, padded_gso


class SNNDataLoader:
    """Data loader wrapper for SNN training"""
    
    def __init__(self, cfg):
        use_pin = torch.cuda.is_available()
        
        # Training dataset
        print("\n📦 Creating training dataset...")
        train_set = SNNDataset(cfg, 'train')
        batch_size = cfg['train'].get('batch_size', 8)
        num_workers = cfg['train'].get('num_workers', 0)
        
        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=use_pin,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=pad_collate_fn  # Use custom collate for variable lengths
        )
        
        print(f"   Train loader: {len(self.train_loader)} batches of {batch_size}")
        
        # Validation dataset
        if 'valid' in cfg:
            print("\n📦 Creating validation dataset...")
            valid_set = SNNDataset(cfg, 'valid')
            valid_batch_size = cfg['valid'].get('batch_size', 4)
            valid_num_workers = cfg['valid'].get('num_workers', 0)
            
            self.valid_loader = DataLoader(
                valid_set,
                batch_size=valid_batch_size,
                shuffle=False,
                pin_memory=use_pin,
                num_workers=valid_num_workers,
                collate_fn=pad_collate_fn  # Use custom collate for variable lengths
            )
            
            print(f"   Valid loader: {len(self.valid_loader)} batches of {valid_batch_size}")
        else:
            self.valid_loader = None
            print("   No validation dataset configured")
