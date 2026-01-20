"""Quick script to check action distribution in expert data."""
import pickle
from collections import Counter

# Load training data
with open('data/FourRooms_train.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Collect all actions
all_actions = []
for traj in train_data[:20]:  # Check first 20 trajectories
    all_actions.extend(traj['actions'])

# Count actions
action_counts = Counter(all_actions)

print(f"Total actions in first 20 trajectories: {len(all_actions)}")
print(f"\nAction distribution:")
print(f"  0 (turn_left):   {action_counts[0]:4d} ({100*action_counts[0]/len(all_actions):5.1f}%)")
print(f"  1 (turn_right):  {action_counts[1]:4d} ({100*action_counts[1]/len(all_actions):5.1f}%)")
print(f"  2 (move_forward): {action_counts[2]:4d} ({100*action_counts[2]/len(all_actions):5.1f}%)")
print(f"  3 (pickup):      {action_counts[3]:4d} ({100*action_counts[3]/len(all_actions):5.1f}%)")
print(f"  4 (drop):        {action_counts[4]:4d} ({100*action_counts[4]/len(all_actions):5.1f}%)")
print(f"  5 (toggle):      {action_counts[5]:4d} ({100*action_counts[5]/len(all_actions):5.1f}%)")
print(f"  6 (done):        {action_counts[6]:4d} ({100*action_counts[6]/len(all_actions):5.1f}%)")

print(f"\nSample trajectory lengths: {[len(traj['actions']) for traj in train_data[:5]]}")
print(f"\nSample actions from first trajectory:")
if train_data:
    print(f"  First 20 actions: {train_data[0]['actions'][:20]}")
    print(f"  Unique actions: {set(train_data[0]['actions'])}")
