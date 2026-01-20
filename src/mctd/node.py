"""
MCTD Tree Node.

Represents a node in the Monte Carlo Tree Diffusion search tree.
Each node stores a partially denoised action sequence in hidden space.
"""
import torch
import numpy as np
from typing import Optional, Dict, Any


class MCTDNode:
    """
    Node in MCTD tree representing a partially denoised action sequence.
    
    Stores:
    - hidden_state: [seq_len, hidden_dim] continuous action representation
    - noise_level: float in [0, 1], how noisy the actions are
    - env_state: dict, the maze state at this point
    - statistics: visits, total reward, Q-value
    """
    def __init__(
        self,
        hidden_state: torch.Tensor,
        noise_level: float,
        env_state: Dict[str, Any],
        parent: Optional['MCTDNode'] = None,
        action_taken: Optional[int] = None,
    ):
        """
        Args:
            hidden_state: [seq_len, hidden_dim] continuous action representation
            noise_level: float in [0, 1], current noise level (1.0 = fully noisy, 0.0 = clean)
            env_state: dict with 'grid' and 'direction' keys
            parent: parent node in tree
            action_taken: discrete action index that led to this node (for guidance scale tracking)
        """
        self.hidden_state = hidden_state  # [L, D]
        self.noise_level = noise_level    # float
        self.env_state = env_state        # dict
        self.parent = parent
        self.action_taken = action_taken  # int or None (guidance scale index)
        
        # Tree statistics
        self.children: list['MCTDNode'] = []
        self.visits = 0
        self.total_reward = 0.0
        self.q_value = 0.0
        
        # For analysis
        self.depth = 0 if parent is None else parent.depth + 1
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)"""
        return len(self.children) == 0
    
    def is_terminal(self) -> bool:
        """Check if we've reached final denoising (t=0)"""
        return self.noise_level <= 0.01
    
    def add_child(self, child: 'MCTDNode'):
        """Add a child node"""
        self.children.append(child)
    
    def update(self, reward: float):
        """
        Update statistics after rollout.
        
        Args:
            reward: reward from simulation
        """
        self.visits += 1
        self.total_reward += reward
        self.q_value = self.total_reward / self.visits
    
    def uct_score(self, exploration_const: float = 1.414) -> float:
        """
        Upper Confidence Bound for Trees (UCT).
        
        UCT = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
        
        where:
        - Q(s,a) is average reward from this node
        - N(s) is parent visits
        - N(s,a) is this node's visits
        - c is exploration constant
        
        Args:
            exploration_const: exploration constant (default √2 ≈ 1.414)
        
        Returns:
            UCT score (float('inf') if unvisited)
        """
        if self.visits == 0:
            return float('inf')  # Prioritize unvisited nodes
        
        if self.parent is None:
            return self.q_value
        
        exploitation = self.q_value
        exploration = exploration_const * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )
        
        return exploitation + exploration
    
    def best_child(self, exploration_const: float = 1.414) -> Optional['MCTDNode']:
        """
        Select child with highest UCT score.
        
        Args:
            exploration_const: exploration constant for UCT
        
        Returns:
            Best child node, or None if no children
        """
        if not self.children:
            return None
        
        return max(self.children, key=lambda c: c.uct_score(exploration_const))
    
    def __repr__(self) -> str:
        return (f"MCTDNode(depth={self.depth}, t={self.noise_level:.2f}, "
                f"visits={self.visits}, Q={self.q_value:.3f}, "
                f"children={len(self.children)})")
