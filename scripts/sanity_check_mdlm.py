"""
FIXED sanity check - use deterministic masking
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader
from src.models.diffusion_policy import DiffusionPolicy
from src.environments.trajectory_dataset import TrajectoryDataset
from src.config import get_experiment_config
import torch.nn.functional as F


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    experiment_config = get_experiment_config()
    config = experiment_config.to_dict()
    
    # Load single trajectory
    import pickle
    from pathlib import Path
    data_path = Path(config["data_dir"]) / f"{config['env_name']}_train.pkl"
    with open(data_path, "rb") as f:
        trajectories = pickle.load(f)
    
    grid_size = 19 if "FourRooms" in config["env_name"] else 8
    
    dataset = TrajectoryDataset(
        [trajectories[0]],  # Single trajectory
        max_seq_len=config["max_seq_len"],
        use_augmentation=False,
        grid_size=grid_size,
    )
    
    # Get the single example
    example = dataset[0]
    actions = example['actions'].unsqueeze(0).to(device)  # [1, seq_len]
    states = {k: v.unsqueeze(0).to(device) for k, v in example['states'].items()}
    length = example['length']
    
    print(f"Trajectory length: {length}")
    print(f"Actions: {actions[0, :length].tolist()}")
    
    # Initialize model
    model = DiffusionPolicy(
        num_actions=config["num_actions"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        grid_size=grid_size,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create FIXED mask - mask positions 2, 5, 10 (or whatever is valid)
    mask_positions = [i for i in [2, 5, 10, 15, 20] if i < length]
    print(f"Fixed mask positions: {mask_positions}")
    
    fixed_mask = torch.zeros(1, config["max_seq_len"], dtype=torch.bool, device=device)
    fixed_mask[0, mask_positions] = True
    
    mask_ratio = torch.tensor([len(mask_positions) / length], device=device)
    
    print(f"\n=== Fixed-mask overfit test ===")
    print(f"Masking {len(mask_positions)} positions out of {length}")
    
    model.train()
    for step in range(500):
        # Create masked input with FIXED mask
        masked_actions = actions.clone()
        masked_actions[fixed_mask] = model.mask_token_id
        
        # Forward
        logits = model(masked_actions, states, mask_ratio)
        
        # Loss only on masked positions
        loss_per_token = F.cross_entropy(
            logits.view(-1, model.num_actions),
            actions.view(-1),
            reduction='none'
        ).view(1, -1)
        
        loss = (loss_per_token * fixed_mask.float()).sum() / fixed_mask.sum()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Compute accuracy on masked positions
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == actions) & fixed_mask
            acc = correct.sum().float() / fixed_mask.sum()
        
        if step % 20 == 0:
            # Show what model predicts vs ground truth at masked positions
            gt_at_mask = actions[0, mask_positions].tolist()
            pred_at_mask = preds[0, mask_positions].tolist()
            print(f"Step {step:03d}: loss={loss.item():.4f}, acc={acc.item():.4f}")
            print(f"    GT:   {gt_at_mask}")
            print(f"    Pred: {pred_at_mask}")
    
    print("\n=== Final check: model should perfectly predict masked positions ===")
    model.eval()
    with torch.no_grad():
        masked_actions = actions.clone()
        masked_actions[fixed_mask] = model.mask_token_id
        logits = model(masked_actions, states, mask_ratio)
        preds = logits.argmax(dim=-1)
        
        print(f"Ground truth at masked positions: {actions[0, mask_positions].tolist()}")
        print(f"Predictions at masked positions:  {preds[0, mask_positions].tolist()}")
        
        correct = (preds[0, mask_positions] == actions[0, mask_positions]).all()
        print(f"Perfect match: {correct.item()}")


if __name__ == "__main__":
    main()