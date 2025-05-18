import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MAPFBaseDataset(Dataset):
    """
    Base dataset for MAPF, handles loading and validation of cases.
    Subclasses should implement the __getitem__ method for specific model input needs.
    """

    def __init__(self, config, mode):
        self.config = config[mode]
        self.dir_path = self.config["root_dir"]
        if mode == "valid":
            self.dir_path = os.path.join(self.dir_path, "val")
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

    def __getitem__(self, index):
        states = torch.from_numpy(self.states[index]).float()  # (time, agents, 2, 5, 5)
        trayec = torch.from_numpy(self.trajectories[index]).float()  # (time, agents)
        gsos = torch.from_numpy(self.gsos[index]).float()  # (time, agents, agents)
        # Do NOT flatten states for SNN
        # states = states.view(states.shape[0], states.shape[1], -1)  # REMOVE THIS LINE
        return states, trayec, gsos


class CreateGNNDataset(MAPFBaseDataset):
    """
    Dataset for GNN/baseline models. Returns (states, trajectories, gsos) per time step.
    States are reshaped to (samples, agents, 2, 5, 5) for GNN input.
    """

    def __init__(self, config, mode):
        super().__init__(config, mode)
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
    def __init__(self, config):
        self.config = config
        train_set = CreateGNNDataset(self.config, "train")
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.config["train"]["batch_size"],
            shuffle=True,
            num_workers=self.config["train"].get("num_workers", 0),
            pin_memory=True,
            drop_last=True
        )
        if "valid" in self.config:
            valid_set = CreateGNNDataset(self.config, "valid")
            self.valid_loader = DataLoader(
                valid_set,
                batch_size=self.config["valid"].get("batch_size", 1),
                shuffle=False,
                num_workers=self.config["valid"].get("num_workers", 0),
                pin_memory=True,
            )
        else:
            self.valid_loader = None


class SNNDataLoader:
    def __init__(self, config):
        self.config = config
        train_set = CreateSNNDataset(self.config, "train")
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.config["train"]["batch_size"],
            shuffle=True,
            num_workers=self.config["train"].get("num_workers", 0),
            pin_memory=True,
            drop_last=True
        )
        if "valid" in self.config:
            valid_set = CreateSNNDataset(self.config, "valid")
            self.valid_loader = DataLoader(
                valid_set,
                batch_size=self.config["valid"].get("batch_size", 1),
                shuffle=False,
                num_workers=self.config["valid"].get("num_workers", 0),
                pin_memory=True,
            )
        else:
            self.valid_loader = None


if __name__ == "__main__":
    # Example config for testing
    config = {
        "train": {
            "root_dir": r"dataset/5_8_28/",
            "mode": "train",
            "max_time": 13,
            "nb_agents": 2,
            "min_time": 13,
            "batch_size": 2,
            "num_workers": 0,
        },
        "valid": {
            "root_dir": r"dataset/5_8_28/",
            "mode": "valid",
            "max_time": 13,
            "nb_agents": 2,
            "min_time": 13,
            "batch_size": 1,
            "num_workers": 0,
        },
    }
    print("Testing GNNDataLoader...")
    gnn_loader = GNNDataLoader(config)
    train_batch = next(iter(gnn_loader.train_loader))
    print(f"GNN train batch: {[x.shape for x in train_batch]}")
    if gnn_loader.valid_loader:
        valid_batch = next(iter(gnn_loader.valid_loader))
        print(f"GNN valid batch: {[x.shape for x in valid_batch]}")
    print("Testing SNNDataLoader...")
    snn_loader = SNNDataLoader(config)
    train_batch = next(iter(snn_loader.train_loader))
    print(f"SNN train batch: {[x.shape for x in train_batch]}")
    if snn_loader.valid_loader:
        valid_batch = next(iter(snn_loader.valid_loader))
        print(f"SNN valid batch: {[x.shape for x in valid_batch]}")
