"""
Monte Carlo Tree Diffusion (MCTD) Search.

Core algorithm for searching over continuous hidden representations
that denoise into discrete action sequences.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from .node import MCTDNode


class HiddenSpaceMCTD:
    """
    Monte Carlo Tree Diffusion in continuous hidden space.
    
    Key innovation: Search over hidden action representations,
    decode to discrete actions only when evaluating reward.
    
    Algorithm:
    1. SELECT: Navigate tree using UCT until leaf
    2. EXPAND: Create children via different denoising strategies (guidance scales)
    3. SIMULATE: Fast rollout (jumpy denoising) to evaluate reward
    4. BACKPROPAGATE: Update statistics up the tree
    """
    def __init__(
        self,
        policy_model,
        env,
        action_encoder,
        num_simulations: int = 50,
        exploration_const: float = 1.414,
        guidance_scales: List[float] = [0.0, 0.5, 1.0],
        sparse_timesteps: List[float] = [1.0, 0.5, 0.2, 0.0],
        denoising_step_size: float = 0.2,
        reward_alpha: float = 0.1,
        reference_path_embed: Optional[torch.Tensor] = None,
        device: str = 'cpu',
        initial_seed: Optional[int] = None,
    ):
        """
        Args:
            policy_model: DiffusionPolicy model
            env: MiniGrid environment
            action_encoder: ActionEncoder for converting actions to embeddings
            num_simulations: Number of MCTS simulations
            exploration_const: UCT exploration constant (default √2 ≈ 1.414)
            guidance_scales: List of guidance scales for expansion [0.0, 0.5, 1.0]
            sparse_timesteps: Timesteps for jumpy denoising [1.0, 0.5, 0.2, 0.0]
            denoising_step_size: Step size for denoising during expansion (default 0.2)
            reward_alpha: Coefficient for cosine similarity reward term (default 0.1)
            reference_path_embed: Optional reference path embedding for reward [seq_len, hidden_dim]
            device: Device to run on ('cpu', 'cuda', 'mps')
        """
        self.model = policy_model.to(device)
        self.model.eval()
        
        self.env = env
        self.action_encoder = action_encoder.to(device)
        self.num_simulations = num_simulations
        self.exploration_const = exploration_const
        self.guidance_scales = guidance_scales
        # Use more timesteps for better accuracy (was [1.0, 0.5, 0.2, 0.0])
        # More timesteps = more accurate rollouts but slightly slower
        if sparse_timesteps is None or len(sparse_timesteps) < 4:
            # Default to more granular schedule if not provided
            sparse_timesteps = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
        self.sparse_timesteps = sorted(sparse_timesteps, reverse=True)  # [1.0, ..., 0.0]
        self.denoising_step_size = denoising_step_size
        self.reward_alpha = reward_alpha
        self.reference_path_embed = reference_path_embed
        self.device = device
        self.initial_seed = initial_seed  # Store seed for consistent environment resets
        
        # Store model parameters for shape inference
        self.hidden_dim = self.model.hidden_dim
        self.max_seq_len = self.model.max_seq_len  # For action sequences
    
    @torch.no_grad()
    def search(
        self,
        initial_state: Dict[str, Any],
        reference_path: Optional[torch.Tensor] = None,
        use_similarity_reward: bool = True,
        use_distance_reward: bool = False,
        distance_reward_scale: float = 0.1
    ) -> Tuple[torch.Tensor, MCTDNode]:
        """
        Run MCTD search from initial state.
        
        Args:
            initial_state: dict with 'grid' and 'direction' keys
            reference_path: Optional reference action sequence [seq_len] for reward
            use_similarity_reward: Whether to use cosine similarity reward term (default True)
        
        Returns:
            best_actions: [seq_len] discrete action sequence
            search_tree: root node (for analysis)
        """
        # Compute reference path embedding if provided and similarity reward is enabled
        if use_similarity_reward and reference_path is not None:
            ref_embed = self.action_encoder(reference_path.unsqueeze(0).to(self.device))[0]  # [seq_len, hidden_dim]
        elif use_similarity_reward and self.reference_path_embed is not None:
            ref_embed = self.reference_path_embed
        else:
            ref_embed = None
        
        # Initialize root with fully masked (noisy) hidden state
        # At t=1.0 (noise_level=1.0), ALL positions should be masked (pure noise)
        # This matches training: at high t, mask ratio → 1.0, so all positions get noise
        # IMPORTANT: Use max_seq_len (action sequence length), NOT num_tokens (state grid size)!
        seq_len = self.max_seq_len  # Action sequence length (e.g., 32), not state grid size (361)
        
        # Initialize with pure noise (all positions masked, no action embeddings)
        # This is correct for t=1.0 - everything should be noise, not action 2 embeddings!
        h_init = torch.randn(
            seq_len,
            self.hidden_dim,
            device=self.device
        )
        # Note: In training, masked positions get noise, unmasked get clean embeddings
        # At t=1.0, mask ratio → 1.0, so all positions are masked → all noise
        # This matches that behavior exactly
        
        # Convert initial state to tensor format if needed
        # Ensure all tensors are on the correct device
        if isinstance(initial_state['grid'], np.ndarray):
            initial_state_tensor = {
                'grid': torch.tensor(initial_state['grid'], dtype=torch.float32, device=self.device).unsqueeze(0),
                'direction': torch.tensor([initial_state['direction']], dtype=torch.long, device=self.device),
            }
        else:
            initial_state_tensor = {
                k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() == 0 else v).to(self.device)
                if isinstance(v, torch.Tensor) else v
                for k, v in initial_state.items()
            }
        
        root = MCTDNode(
            hidden_state=h_init,
            noise_level=1.0,
            env_state=initial_state_tensor,
        )
        
        # Run simulations
        for sim in range(self.num_simulations):
            # 1. Selection
            node = self.select(root)
            
            # 2. Expansion
            if not node.is_terminal():
                children = self.expand(node)
                
                # 3. Simulation on each child
                for child in children:
                    reward = self.simulate(
                        child, 
                        ref_embed if use_similarity_reward else None,
                        use_distance_reward=use_distance_reward,
                        distance_reward_scale=distance_reward_scale
                    )
                    
                    # 4. Backpropagation
                    self.backpropagate(child, reward)
        
        # Extract best trajectory
        best_actions = self.extract_best_trajectory(root)
        
        return best_actions, root
    
    def select(self, root: MCTDNode) -> MCTDNode:
        """
        Selection phase: Navigate tree using UCT.
        
        Keep selecting best child until reaching a leaf node.
        
        Args:
            root: Root node of search tree
        
        Returns:
            Selected leaf node
        """
        node = root
        
        while not node.is_leaf() and not node.is_terminal():
            node = node.best_child(self.exploration_const)
        
        return node
    
    def expand(self, node: MCTDNode) -> List[MCTDNode]:
        """
        Expansion phase: Create children via different denoising strategies.
        
        Meta-actions = different guidance scales.
        Each child represents a different refinement of the action plan.
        
        Args:
            node: Node to expand
        
        Returns:
            List of child nodes
        """
        children = []
        
        # Compute next noise level
        t_next = max(0.0, node.noise_level - self.denoising_step_size)
        
        # Denoise with different guidance scales
        # Note: denoise_step doesn't currently support batched guidance scales,
        # so we do individual calls but this is still efficient due to model caching
        batch_denoised = []
        h_batch = node.hidden_state.unsqueeze(0)  # [1, L, D]
        t_batch = torch.tensor([node.noise_level], device=self.device)  # [1]
        
        for guidance_scale in self.guidance_scales:
            denoised = self.model.denoise_step(h_batch, node.env_state, t_batch, guidance_scale)[0]  # [L, D]
            batch_denoised.append(denoised)
        
        # Create child nodes
        # Ensure env_state is on correct device (copy dict but ensure tensors are on device)
        env_state_on_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in node.env_state.items()
        }
        
        for i, (guidance_scale, h_child) in enumerate(zip(self.guidance_scales, batch_denoised)):
            child = MCTDNode(
                hidden_state=h_child,
                noise_level=t_next,
                env_state=env_state_on_device,  # Same state (we're denoising actions, not executing)
                parent=node,
                action_taken=i,  # Store guidance scale index
            )
            
            node.add_child(child)
            children.append(child)
        
        return children
    
    def simulate(
        self, 
        node: MCTDNode, 
        reference_embed: Optional[torch.Tensor] = None,
        use_distance_reward: bool = False,
        distance_reward_scale: float = 0.1
    ) -> float:
        """
        Simulation phase: Fast rollout to t=0 using jumpy denoising.
        
        Instead of full denoising schedule, jump to sparse timesteps.
        This is the key efficiency trick from Fast-MCTD.
        
        Args:
            node: Node to simulate from
            reference_embed: Optional reference path embedding [seq_len, hidden_dim] for similarity reward
        
        Returns:
            Reward from evaluated trajectory
        """
        h = node.hidden_state.clone()
        state = node.env_state
        t = node.noise_level
        
        # Jumpy denoising: skip to sparse timesteps
        remaining_steps = [ts for ts in self.sparse_timesteps if ts < t]
        
        # Batch denoising for remaining steps
        for t_target in remaining_steps:
            h_batch = h.unsqueeze(0)  # [1, L, D]
            t_batch = torch.tensor([t_target], device=self.device)
            
            h = self.model.denoise_step(h_batch, state, t_batch, guidance_scale=1.0)[0]  # [L, D]
        
        # Decode to discrete actions
        # Use model's action head to get logits
        h_batch = h.unsqueeze(0)  # [1, L, D]
        t_final = torch.tensor([0.0], device=self.device)
        
        logits = self.model.forward(h_batch, state, t_final)  # [1, L, num_actions]
        actions = logits.argmax(dim=-1)[0]  # [L]
        
        # Ensure actions are valid (0 to num_actions-1)
        actions = torch.clamp(actions, 0, self.model.num_actions - 1)
        
        # Truncate to actual sequence length (remove padding)
        # For now, use actions up to max_seq_len
        if len(actions) > self.max_seq_len:
            actions = actions[:self.max_seq_len]
        
        # Evaluate in environment
        reward = self.evaluate_trajectory(
            actions.cpu().numpy(), 
            node.env_state, 
            reference_embed, 
            h,
            use_distance_reward=use_distance_reward,
            distance_reward_scale=distance_reward_scale
        )
        
        return reward
    
    def _get_goal_position(self) -> Optional[Tuple[int, int]]:
        """
        Get goal position from environment.
        
        Returns:
            Goal position (x, y) or None if not found
        """
        if not hasattr(self.env.unwrapped, 'grid'):
            return None
        
        grid = self.env.unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                try:
                    cell = grid.get(x, y)
                    if cell and hasattr(cell, 'type') and cell.type == 'goal':
                        return (x, y)
                except:
                    continue
        return None
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def evaluate_trajectory(
        self,
        actions: np.ndarray,
        initial_state: Dict[str, Any],
        reference_embed: Optional[torch.Tensor] = None,
        sequence_embed: Optional[torch.Tensor] = None,
        use_distance_reward: bool = False,
        distance_reward_scale: float = 0.1
    ) -> float:
        """
        Execute action sequence in environment and return reward.
        
        Reward components:
        - success: +50.0 if reached goal (increased from +10.0)
        - efficiency: -0.05 per step (reduced from -0.1)
        - validity: -2.0 per invalid action
        - distance: +scale * (prev_distance - curr_distance) if use_distance_reward=True
        - similarity: +alpha * cosine_sim(reference_embed, sequence_embed)
        
        Args:
            actions: [seq_len] discrete action sequence
            initial_state: Initial environment state
            reference_embed: Optional reference path embedding [seq_len, hidden_dim]
            sequence_embed: Generated sequence embedding [seq_len, hidden_dim]
            use_distance_reward: If True, add intermediate rewards based on distance to goal
            distance_reward_scale: Scale factor for distance rewards (default 0.1)
        
        Returns:
            Total reward
        """
        # Reset environment to initial state using seed
        # CRITICAL FIX: Use stored seed to ensure consistent initial state
        if self.initial_seed is not None:
            obs, info = self.env.reset(seed=self.initial_seed)
        else:
            obs, info = self.env.reset()
        
        # Get goal position for distance-based rewards
        goal_pos = None
        if use_distance_reward:
            goal_pos = self._get_goal_position()
            if goal_pos is None:
                # Can't use distance rewards if we can't find goal
                use_distance_reward = False
        
        # Get initial position for distance tracking
        prev_distance = None
        if use_distance_reward and hasattr(self.env.unwrapped, 'agent_pos'):
            initial_pos = tuple(self.env.unwrapped.agent_pos)
            prev_distance = self._manhattan_distance(initial_pos, goal_pos)
        
        total_reward = 0.0
        steps = 0
        invalid_count = 0
        distance_reward_total = 0.0
        
        for action in actions:
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(int(action))
            done = terminated or truncated
            
            steps += 1
            total_reward += reward
            
            # Distance-based intermediate reward
            if use_distance_reward and hasattr(self.env.unwrapped, 'agent_pos'):
                curr_pos = tuple(self.env.unwrapped.agent_pos)
                curr_distance = self._manhattan_distance(curr_pos, goal_pos)
                
                if prev_distance is not None:
                    # Reward for getting closer, penalize for getting farther
                    distance_change = prev_distance - curr_distance
                    distance_reward = distance_reward_scale * distance_change
                    distance_reward_total += distance_reward
                    total_reward += distance_reward
                
                prev_distance = curr_distance
            
            # Check for invalid action (reward < -0.5 might indicate invalid)
            if reward < -0.5:
                invalid_count += 1
            
            if done:
                if reward > 0:  # Success (reached goal)
                    total_reward += 50.0  # Increased from 10.0 to make success clearly dominate
                break
        
        # Efficiency penalty: -0.05 per step (reduced from -0.1)
        # This ensures successful paths (even long ones) have positive reward
        efficiency_penalty = 0.05 * steps
        total_reward -= efficiency_penalty
        
        # Invalid action penalty: -2.0 per invalid move
        validity_penalty = 2.0 * invalid_count
        total_reward -= validity_penalty
        
        # Cosine similarity reward (if reference provided)
        if reference_embed is not None and sequence_embed is not None:
            # Handle both length and dimension mismatches
            ref_len, ref_dim = reference_embed.shape
            seq_len, seq_dim = sequence_embed.shape
            
            # Pad/truncate to match length
            if ref_len < seq_len:
                # Pad reference with zeros
                padding = torch.zeros(seq_len - ref_len, ref_dim, 
                                    device=reference_embed.device, 
                                    dtype=reference_embed.dtype)
                ref_padded = torch.cat([reference_embed, padding], dim=0)
            elif ref_len > seq_len:
                # Truncate reference
                ref_padded = reference_embed[:seq_len]
            else:
                ref_padded = reference_embed
            
            # Handle dimension mismatch (e.g., different hidden_dim models)
            # Project to the smaller dimension if they don't match
            if ref_dim != seq_dim:
                # Use min dimension and project both to match
                min_dim = min(ref_dim, seq_dim)
                
                # Truncate or pad reference
                if ref_dim > min_dim:
                    ref_padded = ref_padded[:, :min_dim]
                elif ref_dim < min_dim:
                    pad_dim = min_dim - ref_dim
                    ref_padding = torch.zeros(ref_padded.shape[0], pad_dim,
                                             device=ref_padded.device,
                                             dtype=ref_padded.dtype)
                    ref_padded = torch.cat([ref_padded, ref_padding], dim=1)
                
                # Truncate or pad sequence
                if seq_dim > min_dim:
                    seq_padded = sequence_embed[:, :min_dim]
                elif seq_dim < min_dim:
                    pad_dim = min_dim - seq_dim
                    seq_padding = torch.zeros(seq_len, pad_dim,
                                             device=sequence_embed.device,
                                             dtype=sequence_embed.dtype)
                    seq_padded = torch.cat([sequence_embed, seq_padding], dim=1)
                else:
                    seq_padded = sequence_embed
            else:
                seq_padded = sequence_embed
            
            # Flatten embeddings for cosine similarity
            ref_flat = ref_padded.flatten()  # [L*D]
            seq_flat = seq_padded.flatten()  # [L*D]
            
            # Compute cosine similarity (handle case where embeddings might be all zeros)
            if ref_flat.norm() > 1e-8 and seq_flat.norm() > 1e-8:
                cos_sim = F.cosine_similarity(ref_flat.unsqueeze(0), seq_flat.unsqueeze(0))[0].item()
                # Add similarity reward
                similarity_reward = self.reward_alpha * cos_sim
                total_reward += similarity_reward
        
        return total_reward
    
    def backpropagate(self, node: MCTDNode, reward: float):
        """
        Backpropagation phase: Update statistics up the tree.
        
        Args:
            node: Node to start backpropagation from
            reward: Reward from simulation
        """
        current = node
        
        while current is not None:
            current.update(reward)
            current = current.parent
    
    def extract_best_trajectory(self, root: MCTDNode) -> torch.Tensor:
        """
        Extract best action sequence by following highest Q-value path.
        
        Args:
            root: Root node of search tree
        
        Returns:
            [seq_len] discrete action sequence
        """
        node = root
        
        # Follow best children to terminal node
        while not node.is_terminal():
            if not node.children:
                break
            
            # Select child with best combination of Q-value and visit count (confidence)
            # Prefer high Q-value AND high visits (more reliable)
            # Formula: Q-value * (1 + confidence_bonus * visits)
            # This balances exploitation (high Q) with confidence (high visits)
            confidence_bonus = 0.1
            node = max(node.children, key=lambda c: c.q_value * (1 + confidence_bonus * c.visits) if c.visits > 0 else c.q_value)
        
        # Decode final hidden state to actions
        h_batch = node.hidden_state.unsqueeze(0)  # [1, L, D]
        t_final = torch.tensor([0.0], device=self.device)
        
        logits = self.model.forward(h_batch, node.env_state, t_final)  # [1, L, num_actions]
        actions = logits.argmax(dim=-1)[0]  # [L]
        
        # Ensure actions are valid (0 to num_actions-1)
        actions = torch.clamp(actions, 0, self.model.num_actions - 1)
        
        # Truncate to max_seq_len
        if len(actions) > self.max_seq_len:
            actions = actions[:self.max_seq_len]
        
        return actions
