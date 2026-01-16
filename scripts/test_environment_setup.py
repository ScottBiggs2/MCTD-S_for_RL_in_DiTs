"""
Test script to verify environment setup and shape correctness.
Run this before proceeding with data collection.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import minigrid to register environments
try:
    import minigrid
except ImportError:
    print("ERROR: minigrid package not found. Install with: pip install minigrid")
    sys.exit(1)

import torch
from src.environments.minigrid_wrapper import MazeEnvironment
from src.utils.shape_verification import (
    verify_state_shape,
    print_shape_summary
)


def test_environment_basic():
    """Test basic environment functionality."""
    print("Testing basic environment setup...")
    
    try:
        env = MazeEnvironment(env_name="MiniGrid-Empty-8x8-v0")
        print("✓ Environment created successfully")
        
        # Test reset
        state = env.reset()
        print(f"✓ Environment reset successful")
        print_shape_summary(state)
        
        # Verify state shape
        verify_state_shape(state)
        print("✓ State shape verification passed")
        
        # Test step
        action = 2  # forward
        next_state, reward, done, info = env.step(action)
        print(f"✓ Step successful: reward={reward}, done={done}")
        print_shape_summary(next_state)
        
        verify_state_shape(next_state)
        print("✓ Next state shape verification passed")
        
        env.close()
        print("\n✓ All basic environment tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exception(*sys.exc_info())
        return False


def test_multiple_environments():
    """Test multiple MiniGrid environments."""
    print("\nTesting multiple environments...")
    
    env_names = [
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-FourRooms-v0",
    ]
    
    for env_name in env_names:
        try:
            print(f"\nTesting {env_name}...")
            env = MazeEnvironment(env_name=env_name)
            state = env.reset()
            verify_state_shape(state)
            print(f"✓ {env_name} works correctly")
            env.close()
        except Exception as e:
            print(f"✗ {env_name} failed: {e}")
            return False
    
    print("\n✓ All environment tests passed!")
    return True


def test_state_consistency():
    """Test that state shapes are consistent across steps."""
    print("\nTesting state consistency...")
    
    try:
        env = MazeEnvironment()
        state = env.reset()
        initial_grid_shape = state['grid'].shape
        initial_dir_shape = state['direction'].shape
        
        for i in range(10):
            action = 2  # forward
            state, _, done, _ = env.step(action)
            
            assert state['grid'].shape == initial_grid_shape, \
                f"Grid shape changed: {state['grid'].shape} != {initial_grid_shape}"
            assert state['direction'].shape == initial_dir_shape, \
                f"Direction shape changed: {state['direction'].shape} != {initial_dir_shape}"
            
            if done:
                state = env.reset()
        
        env.close()
        print("✓ State shapes are consistent across steps")
        return True
        
    except Exception as e:
        print(f"✗ State consistency test failed: {e}")
        import traceback
        traceback.print_exception(*sys.exc_info())
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Environment Setup Verification")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_environment_basic()
    all_passed &= test_multiple_environments()
    all_passed &= test_state_consistency()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready for data collection!")
    else:
        print("✗ SOME TESTS FAILED - Please fix issues before proceeding")
    print("=" * 60)
