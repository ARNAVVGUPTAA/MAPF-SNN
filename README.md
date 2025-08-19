# Clean Release: Neuromodulated SNN for MAPF

Minimal subset of the original repository containing only the components required to train the neuromodulated spiking neural network on pre-generated MAPF datasets.

## Contents
- `train_neuromod.py` main training entry
- `config.py` simple YAML loader
- `configs/config_snn.yaml` example configuration
- `models/framework_snn.py` minimal neuromodulated SNN model
- `neuromodulated_loss.py` loss / target matching
- `data_loader.py` dataset + dataloader (expects prepared case folders)
- `optimizer_utils.py` optimizer & scheduler factory

## Expected Data Layout
```
<root_dir>/
  train/
    case_XXXX/
      states.npy                # (T+1, A, 2, 5, 5) using frames 1..T
      trajectory_record.npy      # (A, T, ...) we use [:, :T]
      gso.npy                    # (T, ?, A, A) we take [:T,0] + I
  valid/ (optional same layout)
```
Adjust `root_dir`, `nb_agents`, `min_time`, etc. in the YAML.

## Quick Start
```
python clean_release/train_neuromod.py --config clean_release/configs/config_snn.yaml
```

## Notes
- All visualization, debugging and experimental scripts removed.
- Extend the model or loss as needed for research demonstrations.
