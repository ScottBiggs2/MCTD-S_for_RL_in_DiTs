"""
Lightweight ResNet-style CNN tokenizers for state and action encoding/decoding.

Architecture: State -> CNN Tokenizer -> tokens + pos_embed -> DiT -> tokens -> CNN Tokenizer -> Action
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    """Lightweight ResNet block for CNN tokenizer."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class StateCNNTokenizer(nn.Module):
    """
    Lightweight ResNet-style CNN encoder to tokenize MiniGrid states.
    
    Input: (batch, 7, 7, 3) grid observation
    Output: (batch, num_tokens, hidden_dim) tokenized representation
    
    Architecture:
    - ResNet blocks to extract spatial features
    - Flatten spatial dimensions to create tokens
    - Project to hidden_dim
    """
    def __init__(
        self,
        grid_size=7,
        num_channels=3,
        hidden_dim=128,
        num_tokens=None,  # If None, will be grid_size * grid_size
    ):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        
        # Default: one token per spatial location (7x7 = 49 tokens)
        # Can be reduced with pooling if needed
        if num_tokens is None:
            self.num_tokens = grid_size * grid_size
        else:
            self.num_tokens = num_tokens
        
        # Initial convolution
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # ResNet blocks
        self.layer1 = ResNetBlock(32, 64, stride=1)
        self.layer2 = ResNetBlock(64, 128, stride=1)
        
        # Adaptive pooling if we want fewer tokens
        if self.num_tokens < grid_size * grid_size:
            # Use adaptive pooling to reduce spatial size
            pool_size = int(self.num_tokens ** 0.5)  # Assume square
            self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
            spatial_size = pool_size * pool_size
        else:
            self.pool = nn.Identity()
            spatial_size = grid_size * grid_size
        
        # Project to hidden_dim more efficiently
        # Instead of huge linear layer, use adaptive pooling + smaller projection
        # Reduce channels first, then project to tokens
        self.channel_reduce = nn.Conv2d(128, hidden_dim, kernel_size=1)  # 1x1 conv to reduce channels
        # Now we have [B, hidden_dim, H, W] - can reshape directly or use adaptive pooling
        self.proj = nn.Linear(hidden_dim, hidden_dim)  # Small projection per token
        
        # Reshape to (batch, num_tokens, hidden_dim)
        self.token_dim = hidden_dim
    
    def forward(self, grid_obs):
        """
        Tokenize grid observation.
        
        Supports variable grid sizes (e.g., 7x7, 19x19).
        
        Args:
            grid_obs: (batch, H, W, num_channels) or (batch, num_channels, H, W)
                    H and W can be any size (not just grid_size)
        
        Returns:
            tokens: (batch, num_tokens, hidden_dim)
                   num_tokens = H*W if num_tokens=None, else uses num_tokens
        """
        # Ensure input is contiguous tensor
        if not isinstance(grid_obs, torch.Tensor):
            grid_obs = torch.tensor(grid_obs, dtype=torch.float32)
        
        # Handle both (H, W, C) and (C, H, W) formats
        if grid_obs.dim() == 4:
            # Check if format is (B, H, W, C) or (B, C, H, W)
            B, dim1, dim2, dim3 = grid_obs.shape
            if dim3 == 3:  # (B, H, W, C) format
                # Convert to (B, C, H, W)
                # Make contiguous after permute to avoid view issues
                grid_obs = grid_obs.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
                H, W = dim1, dim2
            elif dim1 == 3:  # (B, C, H, W) format
                # Already correct format - make contiguous just in case
                grid_obs = grid_obs.contiguous()
                H, W = dim2, dim3
            else:
                raise ValueError(
                    f"Unexpected grid format: shape={grid_obs.shape}, "
                    f"expected (B, H, W, 3) or (B, 3, H, W)"
                )
        else:
            raise ValueError(f"Expected 4D input, got {grid_obs.dim()}D with shape {grid_obs.shape}")
        
        # Verify shape before conv: should be (B, 3, H, W)
        assert grid_obs.dim() == 4, f"Expected 4D tensor, got {grid_obs.dim()}D"
        assert grid_obs.shape[1] == 3, f"Expected 3 channels, got {grid_obs.shape[1]} channels"
        assert grid_obs.is_contiguous(), "Grid observation must be contiguous before conv"
        
        # Initial conv
        x = F.relu(self.bn1(self.conv1(grid_obs)))  # (B, 32, H, W)
        
        # ResNet blocks
        x = self.layer1(x)  # (B, 64, H, W)
        x = self.layer2(x)  # (B, 128, H, W)
        
        # Pooling if needed
        x = self.pool(x)  # (B, 128, H', W')
        
        # Reduce channels to hidden_dim
        x = self.channel_reduce(x)  # (B, hidden_dim, H', W')
        
        # Reshape to tokens: [B, hidden_dim, H', W'] -> [B, H'*W', hidden_dim]
        B, C, H, W = x.shape
        # Permute and make contiguous to avoid view issues
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H', W', hidden_dim)
        x = x.reshape(B, H * W, C)  # (B, H'*W', hidden_dim)
        
        # Store actual spatial size for token adjustment
        actual_spatial_size = H * W
        
        # If we have fewer/more tokens than needed, pad/truncate
        if actual_spatial_size != self.num_tokens:
            if actual_spatial_size < self.num_tokens:
                # Pad with zeros
                padding = torch.zeros(B, self.num_tokens - actual_spatial_size, C, device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=1)
            else:
                # Truncate
                x = x[:, :self.num_tokens, :]
        
        # Small projection per token (optional, can be identity)
        x = self.proj(x)  # (B, num_tokens, hidden_dim)
        
        return x


class ActionCNNTokenizer(nn.Module):
    """
    Lightweight ResNet-style CNN decoder to convert tokens back to actions.
    
    Input: (batch, num_tokens, hidden_dim) tokenized representation
    Output: (batch, num_actions) action logits per token position
    
    Architecture:
    - Reshape tokens to spatial grid
    - ResNet blocks to decode
    - Project to action space
    """
    def __init__(
        self,
        hidden_dim=128,
        num_tokens=49,  # grid_size * grid_size
        num_actions=7,
        grid_size=7,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.num_actions = num_actions
        self.grid_size = grid_size
        
        # Determine spatial size
        spatial_size = int(num_tokens ** 0.5)
        if spatial_size * spatial_size != num_tokens:
            # Not a perfect square, use grid_size
            spatial_size = grid_size
        
        # Project tokens to spatial features
        self.proj = nn.Linear(hidden_dim, 128)
        
        # Reshape to spatial grid
        self.spatial_size = spatial_size
        
        # ResNet blocks (reverse of encoder)
        self.layer1 = ResNetBlock(128, 64, stride=1)
        self.layer2 = ResNetBlock(64, 32, stride=1)
        
        # Final projection to action logits
        # Output one action per spatial location, then aggregate
        self.conv_final = nn.Conv2d(32, num_actions, kernel_size=1)
        
        # Global pooling to get single action prediction per token
        # Or we can output per-token actions
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, tokens):
        """
        Decode tokens to action logits.
        
        Args:
            tokens: (batch, num_tokens, hidden_dim)
        
        Returns:
            action_logits: (batch, num_tokens, num_actions) or (batch, num_actions)
        """
        B, num_tokens, D = tokens.shape
        
        # Project tokens
        x = self.proj(tokens)  # (B, num_tokens, 128)
        
        # Reshape to spatial grid
        spatial_size = self.spatial_size
        # Reshape to (B, 128, H, W)
        # Handle case where num_tokens might not match spatial_size^2
        if num_tokens == spatial_size * spatial_size:
            # Perfect match: reshape directly
            x = x.reshape(B, spatial_size, spatial_size, 128)
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, 128, H, W)
        else:
            # Need to pad or crop tokens to match spatial_size^2
            target_size = spatial_size * spatial_size
            if num_tokens > target_size:
                # Crop: take first target_size tokens
                x = x[:, :target_size, :]
            else:
                # Pad: repeat last token or use zeros
                padding_size = target_size - num_tokens
                last_token = x[:, -1:, :].expand(B, padding_size, 128).contiguous()
                x = torch.cat([x, last_token], dim=1)
            
            # Now reshape to spatial grid
            x = x.reshape(B, spatial_size, spatial_size, 128)
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, 128, H, W)
        
        # ResNet blocks
        x = self.layer1(x)  # (B, 64, H, W)
        x = self.layer2(x)  # (B, 32, H, W)
        
        # Project to action logits
        x = self.conv_final(x)  # (B, num_actions, H, W)
        
        # For now, return per-token actions: (B, num_tokens, num_actions)
        # Reshape back
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, num_actions)
        # Flatten spatial dimensions and ensure we have num_tokens
        H, W = x.shape[1], x.shape[2]
        x = x.reshape(B, H * W, self.num_actions)
        
        # If H*W != num_tokens, we need to pad or crop
        if H * W != num_tokens:
            if H * W > num_tokens:
                # Crop to num_tokens
                x = x[:, :num_tokens, :]
            else:
                # Pad with zeros (or repeat last tokens)
                padding = torch.zeros(B, num_tokens - H * W, self.num_actions, 
                                    device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=1)
        
        return x
