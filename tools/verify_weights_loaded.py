#!/usr/bin/env python3
"""Verify that model weights are loaded correctly from checkpoint.

This script:
1. Loads a checkpoint file
2. Builds the model from config
3. Loads the checkpoint into the model (using the same logic as train.py)
4. Verifies that loaded weights match checkpoint weights
5. Reports statistics on matched/missing/unexpected keys
"""

import argparse
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mmengine import Config
from coopdet3d.models import build_coop_model


def filter_camera_keys(state_dict):
    """Filter out camera encoder keys (same logic as train.py)."""
    camera_patterns = [
        'nodes.vehicle.fusion_model.encoders.camera.backbone',
        'nodes.infrastructure.fusion_model.encoders.camera.backbone',
        'nodes.vehicle.fusion_model.encoders.camera.neck',
        'nodes.infrastructure.fusion_model.encoders.camera.neck',
        'nodes.vehicle.fusion_model.encoders.camera.vtransform',
        'nodes.infrastructure.fusion_model.encoders.camera.vtransform',
    ]
    
    filtered_state_dict = {}
    camera_keys_removed = []
    
    for key, value in state_dict.items():
        is_camera_key = any(pattern in key for pattern in camera_patterns)
        if is_camera_key:
            camera_keys_removed.append(key)
        else:
            filtered_state_dict[key] = value
    
    return filtered_state_dict, camera_keys_removed


def verify_weights_loaded(config_path, checkpoint_path, strict=False, verbose=False):
    """Verify that checkpoint weights are correctly loaded into model.
    
    Args:
        config_path: Path to model config file
        checkpoint_path: Path to checkpoint file
        strict: Whether to use strict loading (default: False, matching train.py)
        verbose: Whether to print detailed information about each key
    
    Returns:
        dict with verification results
    """
    print(f"\n{'='*80}")
    print("WEIGHT LOADING VERIFICATION")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return None
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' not in checkpoint:
        print("ERROR: Checkpoint does not contain 'state_dict' key")
        print(f"Available keys: {list(checkpoint.keys())}")
        return None
    
    checkpoint_state_dict = checkpoint['state_dict']
    print(f"Checkpoint contains {len(checkpoint_state_dict)} keys")
    
    # Filter camera keys (same as train.py)
    filtered_checkpoint, camera_keys_removed = filter_camera_keys(checkpoint_state_dict)
    if camera_keys_removed:
        print(f"Filtered out {len(camera_keys_removed)} camera encoder keys "
              f"(loaded from YOLOv8 via init_cfg)")
    
    print(f"Checkpoint keys after filtering: {len(filtered_checkpoint)}")
    
    # Build model
    print("\nBuilding model from config...")
    cfg = Config.fromfile(str(config_path))
    model = build_coop_model(cfg.model)
    
    # Get model state dict
    model_state_dict = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())
    model_all = {**model_state_dict, **model_buffers}
    
    print(f"Model has {len(model_state_dict)} parameters and {len(model_buffers)} buffers")
    print(f"Total model keys: {len(model_all)}")
    
    # Load checkpoint into model (same as train.py)
    print("\nLoading checkpoint into model...")
    missing_keys, unexpected_keys = model.load_state_dict(filtered_checkpoint, strict=strict)
    
    print(f"Loading completed (strict={strict})")
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Verify loaded weights match checkpoint
    print("\n" + "="*80)
    print("VERIFICATION: Comparing loaded weights to checkpoint")
    print("="*80)
    
    matched_keys = []
    shape_mismatch_keys = []
    value_mismatch_keys = []
    missing_in_model = []
    unexpected_in_checkpoint = []
    
    # Check all checkpoint keys
    for key in filtered_checkpoint.keys():
        if key not in model_all:
            missing_in_model.append(key)
        else:
            checkpoint_tensor = filtered_checkpoint[key]
            model_tensor = model_all[key]
            
            # Check shape
            if checkpoint_tensor.shape != model_tensor.shape:
                shape_mismatch_keys.append((key, checkpoint_tensor.shape, model_tensor.shape))
            else:
                # Check values
                if torch.allclose(checkpoint_tensor, model_tensor, rtol=1e-5, atol=1e-8):
                    matched_keys.append(key)
                else:
                    max_diff = (checkpoint_tensor - model_tensor).abs().max().item()
                    value_mismatch_keys.append((key, max_diff))
    
    # Check for unexpected keys in checkpoint
    for key in filtered_checkpoint.keys():
        if key not in model_all:
            unexpected_in_checkpoint.append(key)
    
    # Print results
    print(f"\n✓ Matched keys: {len(matched_keys)}")
    print(f"✗ Shape mismatch: {len(shape_mismatch_keys)}")
    print(f"✗ Value mismatch: {len(value_mismatch_keys)}")
    print(f"✗ Missing in model: {len(missing_in_model)}")
    print(f"⚠ Unexpected in checkpoint: {len(unexpected_in_checkpoint)}")
    
    # Detailed information
    if shape_mismatch_keys and verbose:
        print(f"\n{'='*80}")
        print("SHAPE MISMATCHES:")
        print("="*80)
        for key, ckpt_shape, model_shape in shape_mismatch_keys[:10]:
            print(f"  {key}")
            print(f"    Checkpoint: {ckpt_shape}")
            print(f"    Model:      {model_shape}")
        if len(shape_mismatch_keys) > 10:
            print(f"  ... and {len(shape_mismatch_keys) - 10} more")
    
    if value_mismatch_keys and verbose:
        print(f"\n{'='*80}")
        print("VALUE MISMATCHES (max difference):")
        print("="*80)
        for key, max_diff in sorted(value_mismatch_keys, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {max_diff:.2e}")
        if len(value_mismatch_keys) > 10:
            print(f"  ... and {len(value_mismatch_keys) - 10} more")
    
    if missing_in_model and verbose:
        print(f"\n{'='*80}")
        print("MISSING IN MODEL (first 20):")
        print("="*80)
        for key in missing_in_model[:20]:
            print(f"  {key}")
        if len(missing_in_model) > 20:
            print(f"  ... and {len(missing_in_model) - 20} more")
    
    if unexpected_in_checkpoint and verbose:
        print(f"\n{'='*80}")
        print("UNEXPECTED IN CHECKPOINT (first 20):")
        print("="*80)
        for key in unexpected_in_checkpoint[:20]:
            print(f"  {key}")
        if len(unexpected_in_checkpoint) > 20:
            print(f"  ... and {len(unexpected_in_checkpoint) - 20} more")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    total_checkpoint_keys = len(filtered_checkpoint)
    total_matched = len(matched_keys)
    match_rate = (total_matched / total_checkpoint_keys * 100) if total_checkpoint_keys > 0 else 0
    
    print(f"Checkpoint keys (after filtering): {total_checkpoint_keys}")
    print(f"Successfully matched: {total_matched} ({match_rate:.1f}%)")
    print(f"Issues found: {len(shape_mismatch_keys) + len(value_mismatch_keys) + len(missing_in_model)}")
    
    if len(shape_mismatch_keys) == 0 and len(value_mismatch_keys) == 0 and len(missing_in_model) == 0:
        print("\n✓ SUCCESS: All checkpoint weights loaded correctly!")
    else:
        print("\n⚠ WARNING: Some weights may not have loaded correctly")
        if len(shape_mismatch_keys) > 0:
            print(f"  - {len(shape_mismatch_keys)} keys have shape mismatches")
        if len(value_mismatch_keys) > 0:
            print(f"  - {len(value_mismatch_keys)} keys have value mismatches")
        if len(missing_in_model) > 0:
            print(f"  - {len(missing_in_model)} keys are missing in model")
    
    # Sample matched keys
    if matched_keys and verbose:
        print(f"\n{'='*80}")
        print("SAMPLE MATCHED KEYS (first 10):")
        print("="*80)
        for key in matched_keys[:10]:
            print(f"  ✓ {key}")
    
    return {
        'matched': len(matched_keys),
        'shape_mismatch': len(shape_mismatch_keys),
        'value_mismatch': len(value_mismatch_keys),
        'missing': len(missing_in_model),
        'unexpected': len(unexpected_in_checkpoint),
        'total_checkpoint': total_checkpoint_keys,
        'match_rate': match_rate,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify that model weights are loaded correctly from checkpoint"
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to model config file'
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Use strict loading (default: False, matching train.py behavior)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed information about mismatches'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    
    results = verify_weights_loaded(
        str(config_path),
        str(checkpoint_path),
        strict=args.strict,
        verbose=args.verbose
    )
    
    if results is None:
        sys.exit(1)
    
    # Exit with error if there are issues
    if results['shape_mismatch'] > 0 or results['value_mismatch'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

