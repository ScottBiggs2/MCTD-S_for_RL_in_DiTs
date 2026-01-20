"""
VAE-style CNN encoder for state conditioning with residual connections.
Outputs a single embedding vector from the grid image via latent bottleneck.

Supports pretraining with:
- Grid reconstruction (decoder)
- Agent position prediction
- Goal position prediction
- Direction prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ResNetBlock(nn.Module):
    """
    Simple ResNet block: 3 convolutions with residual addition.
    
    Architecture:
    - conv1: changes channels/stride
    - conv2: keeps channels
    - Add: input + conv2 output (dimensions must match)
    - conv3: output convolution (changes channels)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # First conv: changes channels and/or stride
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Second conv: keeps channels (must match input for residual)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        # Third conv: output (changes channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Projection for residual if spatial dimensions change (stride != 1)
        self.use_residual = (stride == 1)

    
    def forward(self, x):
        # First conv + BN + ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv + BN
        out = self.bn2(self.conv2(out))
        
        # Residual: add input only when dimensions match exactly
        if self.use_residual:
            out = F.relu(out + x)
        else:
            out = F.relu(out)
        
        # Third conv (output - changes channels)
        out = F.relu(self.bn3(self.conv3(out)))
        return out


class ResNetDecoderBlock(nn.Module):
    """
    Simple ResNet decoder block: 3 convolutions with residual addition.
    
    Architecture:
    - conv1: upsamples (if stride > 1), keeps channels
    - conv2: keeps channels
    - Add: input + conv2 output (dimensions must match)
    - conv3: output convolution (changes channels)
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        # First conv: upsamples (if stride > 1), keeps channels for residual
        # Use output_padding to ensure exact dimension matching
        if stride > 1:
            self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=stride, padding=1, output_padding=stride-1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Second conv: keeps channels (must match for residual)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        # Third conv: output (changes channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Only use residual connection when dimensions match exactly (stride == 1)
        self.use_residual = (stride == 1)
    
    def forward(self, x):
        # First conv + BN + ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv + BN
        out = self.bn2(self.conv2(out))

        # Residual: add input only when dimensions match exactly
        if self.use_residual:
            out = F.relu(out + x)
        else:
            out = F.relu(out)

        # Third conv (output - changes channels)
        out = F.relu(self.bn3(self.conv3(out)))
        return out


class StateCNNEncoder(nn.Module):
    """
    Lightweight CNN that encodes grid observation to a single embedding vector.
    
    CLIP-style: Image -> CNN -> Single vector embedding
    
    Input: [B, H, W, 3] grid observation (or [B, H*W*3] flattened)
    Output: [B, hidden_dim] state embedding
    
    Can be pretrained with reconstruction and auxiliary tasks.
    """
    def __init__(
        self,
        grid_size=19,  # For FourRooms: 19x19
        hidden_dim=128,
        num_channels=3,
        enable_decoder: bool = False,  # Enable decoder for pretraining
        enable_auxiliary: bool = False,  # Enable auxiliary prediction heads
    ):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.enable_decoder = enable_decoder
        self.enable_auxiliary = enable_auxiliary
        
        # ResNet-style encoder with residual connections (reduced channels for efficiency)
        # Initial conv
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # H/4, W/4
        
        # ResNet blocks with residual connections (reduced channels)
        self.layer1 = self._make_layer(32, 64, 2, stride=1)   # H/4, W/4
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # H/8, W/8
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # H/16, W/16
        
        # Global Average Pooling: [B, 256, H/16, W/16] -> [B, 256]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Latent projection (no VAE sampling - deterministic)
        self.latent_dim = hidden_dim
        self.fc_mu = nn.Linear(256, self.latent_dim)  # Reduced from 512
        
        # Project latent to hidden_dim (with residual-style connection)
        self.latent_proj = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Optional: direction embedding
        self.direction_embed = nn.Embedding(4, hidden_dim)
        self.direction_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # For concatenating direction
        
        # Decoder for reconstruction (only if enabled) - symmetric to encoder with residuals
        if enable_decoder:
            # Decoder: latent -> features -> reconstruction
            # Project latent back to feature space (reduced channels)
            self.decoder_fc = nn.Linear(self.latent_dim, 256 * (grid_size // 16) * (grid_size // 16))
            self.decoder_reshape_size = (grid_size // 16, grid_size // 16)
            
            # ResNet-style decoder blocks (reverse of encoder, reduced channels)
            # Note: stride=2 means upsample (double spatial size)
            # Each layer: upsample (if stride=2) then change channels in conv3
            self.decoder_layer3 = self._make_decoder_layer(256, 128, 2, stride=2)  # H/16 -> H/8, 256->128
            self.decoder_layer2 = self._make_decoder_layer(128, 64, 2, stride=2)  # H/8 -> H/4, 128->64
            self.decoder_layer1 = self._make_decoder_layer(64, 32, 2, stride=1)   # H/4 -> H/4, 64->32
            
            # Final upsampling layers
            # Use output_padding=1 for stride=2 to ensure exact dimension matching
            self.decoder_upsample1 = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
            )
            self.decoder_upsample2 = nn.Sequential(
                nn.ConvTranspose2d(16, num_channels, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False),
                nn.Sigmoid(),  # Normalize to [0, 1]
            )
        
        # Store target size for interpolation
        self.target_size = grid_size
        
        # Auxiliary prediction heads (only if enabled)
        if enable_auxiliary:
            # Agent position prediction: (x, y) coordinates
            self.agent_pos_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2),  # (x, y)
            )
            
            # Goal position prediction: (x, y) coordinates
            self.goal_pos_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2),  # (x, y)
            )
            
            # Direction prediction: 4 classes (0-3)
            self.direction_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 4),  # 4 directions
            )
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """Create a ResNet layer with residual blocks."""
        layers = []
        # First block may have stride > 1
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        # Remaining blocks have stride=1
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _make_decoder_layer(self, in_channels, out_channels, num_blocks, stride=2):
        """Create a decoder ResNet layer (upsampling with residual connections)."""
        layers = []
        # First block upsamples (if stride > 1) or just changes channels
        layers.append(ResNetDecoderBlock(in_channels, out_channels, stride))
        # Remaining blocks keep same size (stride=1) and channels
        for _ in range(1, num_blocks):
            layers.append(ResNetDecoderBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode state to a single embedding vector.
        
        Args:
            state_dict: Dictionary with:
                - 'grid': [B, H, W, 3] or [B, 3, H, W] or [B, seq_len, H*W*3] or [B, H*W*3]
                - 'direction': [B] or [B, seq_len] or [B, 1] agent direction (0-3)
        
        Returns:
            state_embed: [B, hidden_dim] state embedding vector
        """
        grid = state_dict['grid']
        direction = state_dict['direction']
        
        B = grid.shape[0]
        seq_len = None
        
        # Handle different input formats
        if grid.dim() == 4:
            # [B, H, W, 3] or [B, 3, H, W] - already spatial format
            # Check if channels are first or last
            if grid.shape[1] == 3:
                # [B, 3, H, W] - channels first, already correct for CNN
                pass  # Already in correct format
            else:
                # [B, H, W, 3] - channels last, convert to channels first
                grid = grid.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        elif grid.dim() == 3:
            # [B, seq_len, H*W*3] - take first state and reshape
            B, seq_len, flat_size = grid.shape
            grid = grid[:, 0, :].contiguous()  # [B, H*W*3]
            # Reshape to spatial format
            inferred_size = int((flat_size // 3) ** 0.5)
            if inferred_size * inferred_size * 3 != flat_size:
                raise ValueError(f"Cannot infer grid size from flat_size={flat_size}")
            grid = grid.reshape(B, inferred_size, inferred_size, 3)
            grid = grid.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        elif grid.dim() == 2:
            # [B, H*W*3] - already single flattened state
            flat_size = grid.shape[-1]
            inferred_size = int((flat_size // 3) ** 0.5)
            if inferred_size * inferred_size * 3 != flat_size:
                raise ValueError(f"Cannot infer grid size from flat_size={flat_size}")
            grid = grid.reshape(B, inferred_size, inferred_size, 3)
            grid = grid.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        else:
            raise ValueError(f"Unexpected grid shape: {grid.shape}, expected 2D, 3D, or 4D")
        
        # At this point, grid should be [B, 3, H, W] - ready for CNN
        
        # ResNet encoder with residual connections
        x = self.conv1(grid)  # [B, 32, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B, 32, H/4, W/4]
        
        x = self.layer1(x)  # [B, 64, H/4, W/4]
        x = self.layer2(x)  # [B, 128, H/8, W/8]
        x = self.layer3(x)  # [B, 256, H/16, W/16]
        
        # Global average pooling
        x = self.global_pool(x)  # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]
        
        # Simple projection (no VAE sampling)
        z = self.fc_mu(x)  # Direct projection to latent_dim [B, latent_dim]
        
        # Project latent to hidden_dim
        x = self.latent_proj(z)  # [B, hidden_dim]
        
        # Handle direction - take first if sequence
        if direction.dim() == 2:
            direction = direction[:, 0]  # [B]
        elif direction.dim() == 1 and len(direction) == B:
            direction = direction  # [B]
        elif direction.dim() == 1 and len(direction) == B * seq_len if seq_len is not None else B:
            if seq_len is not None:
                direction = direction.reshape(B, seq_len)[:, 0]  # [B]
            else:
                direction = direction  # [B]
        elif direction.dim() == 1:
            # Handle case where direction might be wrong shape - take first B elements
            direction = direction[:B]  # [B]
        
        # CRITICAL: Add direction information - the model MUST know which direction it's facing!
        # Without this, action 2 (move_forward) has no meaning - forward depends on facing direction!
        dir_emb = self.direction_embed(direction)  # [B, hidden_dim]
        # Combine grid encoding with direction encoding
        x = self.direction_proj(torch.cat([x, dir_emb], dim=1))  # [B, hidden_dim]
        
        return x  # [B, hidden_dim]
    
    def forward_with_features(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns latent for reconstruction.
        
        Returns:
            embedding: [B, hidden_dim] state embedding
            latent: [B, latent_dim] latent vector for reconstruction
        """
        grid = state_dict['grid']
        direction = state_dict['direction']
        
        B = grid.shape[0]
        
        # Handle different input formats (same as forward)
        if grid.dim() == 4:
            if grid.shape[1] == 3:
                pass
            else:
                grid = grid.permute(0, 3, 1, 2).contiguous()
        elif grid.dim() == 3:
            B, seq_len, flat_size = grid.shape
            grid = grid[:, 0, :].contiguous()
            inferred_size = int((flat_size // 3) ** 0.5)
            grid = grid.reshape(B, inferred_size, inferred_size, 3)
            grid = grid.permute(0, 3, 1, 2).contiguous()
        elif grid.dim() == 2:
            flat_size = grid.shape[-1]
            inferred_size = int((flat_size // 3) ** 0.5)
            grid = grid.reshape(B, inferred_size, inferred_size, 3)
            grid = grid.permute(0, 3, 1, 2).contiguous()
        
        # ResNet encoder
        x = self.conv1(grid)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Simple projection (no VAE sampling)
        z = self.fc_mu(x)  # Use fc_mu as direct projection to latent_dim [B, latent_dim]
        
        # Project latent to embedding
        embedding = self.latent_proj(z)  # [B, hidden_dim]
        
        # Handle direction
        if direction.dim() == 2:
            direction = direction[:, 0]
        elif direction.dim() == 1 and len(direction) != B:
            direction = direction[:B]
        
        dir_emb = self.direction_embed(direction)
        embedding = self.direction_proj(torch.cat([embedding, dir_emb], dim=1))
        
        return embedding, z  # Return embedding and latent for reconstruction
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector back to grid image.
        
        Args:
            latent: [B, latent_dim] latent vector
            
        Returns:
            reconstructed_grid: [B, 3, H, W] reconstructed grid
        """
        if not self.enable_decoder:
            raise RuntimeError("Decoder not enabled. Set enable_decoder=True in __init__")
        
        B = latent.shape[0]
        
        # Project latent back to feature space
        x = self.decoder_fc(latent)  # [B, 256 * H/16 * W/16]
        h, w = self.decoder_reshape_size
        x = x.view(B, 256, h, w)  # [B, 256, H/16, W/16]
        
        # ResNet decoder blocks with residual connections
        x = self.decoder_layer3(x)  # [B, 128, H/8, W/8]
        x = self.decoder_layer2(x)  # [B, 64, H/4, W/4]
        x = self.decoder_layer1(x)  # [B, 32, H/4, W/4]
        
        # Final upsampling
        x = self.decoder_upsample1(x)  # [B, 16, H/2, W/2]
        x = self.decoder_upsample2(x)  # [B, 3, H, W]
        
        # Ensure exact size match by interpolating to target size
        x = torch.nn.functional.interpolate(
            x, 
            size=(self.target_size, self.target_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        return x
    
    def predict_auxiliary(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict auxiliary targets from embedding.
        
        Args:
            embedding: [B, hidden_dim] state embedding
            
        Returns:
            Dictionary with:
            - 'agent_pos': [B, 2] predicted agent position (x, y)
            - 'goal_pos': [B, 2] predicted goal position (x, y)
            - 'direction': [B, 4] predicted direction logits
        """
        if not self.enable_auxiliary:
            raise RuntimeError("Auxiliary heads not enabled. Set enable_auxiliary=True in __init__")
        
        return {
            'agent_pos': self.agent_pos_head(embedding),
            'goal_pos': self.goal_pos_head(embedding),
            'direction': self.direction_head(embedding),
        }
