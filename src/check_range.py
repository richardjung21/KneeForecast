
import argparse
import torch
from dataset.knee import KneeXRDataset
from tqdm import tqdm

def check_data_range():
    # Mock args
    class Args:
        data_dir = "/mnt/sdg2/seungho/Dataset/Forecasting/list_dict_reduced.pkl"
        target_dim = (256, 256)
        binary = False
        split_by_site = True
    
    args = Args()
    
    dataset = KneeXRDataset(pickle_dir=args.data_dir, target_dim=args.target_dim, split='train', binary=args.binary, split_by_site=args.split_by_site)
    
    regression_classes = ["MCMJSW", "TPCFDS"]
    min_vals = {k: float('inf') for k in regression_classes}
    max_vals = {k: float('-inf') for k in regression_classes}
    
    print(f"Checking {len(dataset)} samples...")
    
    # Check a subset to be fast
    for i in range(min(1000, len(dataset))):
        _, _, metadata = dataset[i]
        for key in regression_classes:
            val = metadata[key]
            if isinstance(val, torch.Tensor):
                val = val.item()
            if val < min_vals[key]:
                min_vals[key] = val
            if val > max_vals[key]:
                max_vals[key] = val
                
    print("Regression target ranges:")
    for key in regression_classes:
        print(f"{key}: min={min_vals[key]}, max={max_vals[key]}")

if __name__ == "__main__":
    check_data_range()
