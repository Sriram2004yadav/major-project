import json
from pathlib import Path
import re

# --- CONFIG ---
DATA_ROOT = Path("./WUSU_processed")
SITE_NAMES = ["HS", "JA"] 
# --- CONFIG ---

def generate_wusu_samples(run_type):
    """
    Generates the sample list by scanning the CHANGE DETECTION folder, 
    as those files reliably contain the sample index.
    """
    samples = []
    
    # WUSU uses 'train' and 'test' as top-level folders. 'val' is a subset of 'train'.
    dataset_folder = 'train' if run_type in ['train', 'val'] else 'test'
    
    for site in SITE_NAMES:
        # We will scan the 'change/BCD' folder for 2015-2016 change masks to get the index.
        change_dir = DATA_ROOT / dataset_folder / site / 'change' / 'BCD'
        
        if not change_dir.exists():
            print(f"Warning: Directory not found: {change_dir}")
            continue

        # Look for the change files, which are named like 'HS1516_0.tif'
        for file_path in change_dir.glob(f'{site}1516_*.tif'):
            
            # The stem is the filename without the extension (e.g., 'HS1516_0')
            stem = file_path.stem 
            
            # Use regex to find the index at the end of the stem (e.g., '0' from 'HS1516_0')
            match = re.search(r'_(\d+)$', stem) 
            if match:
                index_str = match.group(1) 
            else:
                continue 
            
            # Append the sample structure
            samples.append({
                'site': site,
                'index': index_str,
            })
            
    return samples

# --- EXECUTION ---
# Get all training/validation samples
all_train_samples = generate_wusu_samples('train')
all_test_samples = generate_wusu_samples('test')

# Combine all samples into one list (the original code used this approach)
final_train_json = [{"site": s['site'], "index": s['index'], "split": "train"} for s in all_train_samples]
final_test_json = [{"site": s['site'], "index": s['index'], "split": "test"} for s in all_test_samples]


# Write the files
with open(DATA_ROOT / "samples_train.json", "w") as f:
    json.dump(final_train_json, f, indent=4)
    
with open(DATA_ROOT / "samples_test.json", "w") as f:
    json.dump(final_test_json, f, indent=4)
    
print(f"\nSuccessfully generated {len(final_train_json)} training/validation sample entries.")
print(f"Successfully generated {len(final_test_json)} test sample entries.")
print("\nJSON files created in WUSU_processed/. You must now manually assign a subset of samples_train.json to 'split': 'val'.")