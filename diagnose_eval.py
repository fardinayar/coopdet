#!/usr/bin/env python
"""Diagnose why BEV/3D mAP are low despite good center-based metrics."""

import pickle
import json
import numpy as np
import sys
import os
import glob
from datetime import datetime

def find_results_files():
    """Find all results_nusc.json files in common locations."""
    results_files = []
    
    # Direct file path if provided
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
        if os.path.exists(results_file):
            return [results_file]
        else:
            print(f"Error: Results file not found: {results_file}")
            sys.exit(1)
    
    # Search patterns (in order of preference)
    search_patterns = [
        # Temporary directories
        '/tmp/*/results/results_nusc.json',
        '/tmp/**/results_nusc.json',
        # Work directories (common structure)
        'work_dirs/*/results/results_nusc.json',
        'work_dirs/*/*/results/results_nusc.json',
        'work_dirs/**/results_nusc.json',
        # Current directory and subdirectories
        './results/results_nusc.json',
        './**/results_nusc.json',
        # Absolute path from workspace
        '/work/gn21/h62001/coopdet/work_dirs/**/results_nusc.json',
        '/work/gn21/h62001/coopdet/**/results_nusc.json',
    ]
    
    # Use os.walk for more reliable recursive search
    base_dirs = [
        '/tmp',
        'work_dirs',
        '.',
        '/work/gn21/h62001/coopdet/work_dirs',
    ]
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                if 'results_nusc.json' in files:
                    full_path = os.path.join(root, 'results_nusc.json')
                    if full_path not in results_files:
                        results_files.append(full_path)
    
    # Also try glob patterns
    for pattern in search_patterns:
        try:
            if '**' in pattern:
                found = glob.glob(pattern, recursive=True)
            else:
                found = glob.glob(pattern)
            for f in found:
                if f not in results_files and os.path.exists(f):
                    results_files.append(f)
        except Exception:
            continue
    
    return results_files

# Find results files
results_files = find_results_files()

if not results_files:
    print("No results file found!")
    print("\nThe script searched for 'results_nusc.json' in:")
    print("  - /tmp/*/results/")
    print("  - work_dirs/*/results/")
    print("  - work_dirs/*/*/results/")
    print("  - Current directory and subdirectories")
    print("\nTo use this script:")
    print("  1. Run evaluation first to generate results_nusc.json")
    print("  2. Or provide the path directly: python diagnose_eval.py <path_to_results_nusc.json>")
    print("\nResults are typically saved when running validation/evaluation.")
    sys.exit(1)

# Use most recently modified file
results_file = max(results_files, key=os.path.getmtime)
if len(results_files) > 1:
    print(f"Found {len(results_files)} results file(s), using most recent:")
    for f in sorted(results_files, key=os.path.getmtime, reverse=True)[:5]:
        mtime = os.path.getmtime(f)
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  - {f} (modified: {mtime_str})")
    print(f"\nUsing: {results_file}")
else:
    print(f"Found results file: {results_file}")

print(f"Loading predictions from: {results_file}")

with open(results_file) as f:
    pred_data = json.load(f)

# Load GT
with open('data/tumtraf_v2x_cooperative_perception_dataset_processed/tumtraf_v2x_nusc_infos_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

# Analyze first sample
sample_idx = 0
sample = val_data['infos'][sample_idx]
timestamp = sample['timestamp']

print(f"\n{'='*70}")
print(f"Sample {sample_idx} (timestamp: {timestamp})")
print(f"{'='*70}")

# GT boxes
gt_boxes = sample['gt_boxes']
gt_names = sample['gt_names']

print(f"\nGround Truth: {len(gt_boxes)} boxes")
for i, (box, name) in enumerate(list(zip(gt_boxes, gt_names))[:3]):
    print(f"  {name}: center_z={box[2]:.2f}, dims=[{box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f}]")

# Predictions
# Results are keyed by timestamp, not sample index
pred_boxes_list = pred_data['results'].get(str(timestamp), [])
if not pred_boxes_list:
    # Try sample index as fallback
    pred_boxes_list = pred_data['results'].get(str(sample_idx), [])
print(f"\nPredictions: {len(pred_boxes_list)} boxes")

for i, pred in enumerate(pred_boxes_list[:3]):
    print(f"  {pred['detection_name']}: ")
    print(f"    center_z={pred['translation'][2]:.2f}")
    print(f"    dims={pred['size']}")
    print(f"    score={pred.get('detection_score', 'N/A')}")

# Check if z-coordinates match
if len(pred_boxes_list) > 0 and len(gt_boxes) > 0:
    print(f"\n{'='*70}")
    print("Z-coordinate comparison:")
    print(f"{'='*70}")
    print(f"GT z-coord range: [{gt_boxes[:, 2].min():.2f}, {gt_boxes[:, 2].max():.2f}]")

    pred_z = [p['translation'][2] for p in pred_boxes_list]
    print(f"Pred z-coord range: [{min(pred_z):.2f}, {max(pred_z):.2f}]")
    print(f"Z-coord difference: {abs(np.mean(gt_boxes[:, 2]) - np.mean(pred_z)):.2f}m")

    # Check dimension differences
    print(f"\n{'='*70}")
    print("Dimension comparison (first 3 boxes of same class):")
    print(f"{'='*70}")

    for gt_box, gt_name in zip(gt_boxes[:3], gt_names[:3]):
        # Find matching prediction
        matching_preds = [p for p in pred_boxes_list if p['detection_name'] == gt_name]
        if matching_preds:
            pred = matching_preds[0]
            gt_dims = gt_box[3:6]
            pred_dims = np.array(pred['size'])

            print(f"\n{gt_name}:")
            print(f"  GT dims:   [{gt_dims[0]:.2f}, {gt_dims[1]:.2f}, {gt_dims[2]:.2f}]")
            print(f"  Pred dims: [{pred_dims[0]:.2f}, {pred_dims[1]:.2f}, {pred_dims[2]:.2f}]")
            print(f"  Diff:      [{abs(gt_dims[0]-pred_dims[0]):.2f}, {abs(gt_dims[1]-pred_dims[1]):.2f}, {abs(gt_dims[2]-pred_dims[2]):.2f}]")
            print(f"  % Error:   [{abs(gt_dims[0]-pred_dims[0])/gt_dims[0]*100:.1f}%, {abs(gt_dims[1]-pred_dims[1])/gt_dims[1]*100:.1f}%, {abs(gt_dims[2]-pred_dims[2])/gt_dims[2]*100:.1f}%]")
