import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- CONFIG ---
DATA_ROOT = Path("./WUSU_processed")
TRAIN_JSON_FILE = DATA_ROOT / "samples_train.json"
# The original paper used 90% train / 10% val
VAL_SPLIT_PERCENTAGE = 0.10 
# Set a fixed random state so the split is always the same (reproducibility)
RANDOM_STATE = 42
# --- CONFIG ---

def create_reproducible_val_split():
    if not TRAIN_JSON_FILE.exists():
        print(f"Error: {TRAIN_JSON_FILE} not found. Did you run create_wusu_jsons.py first?")
        return

    with open(TRAIN_JSON_FILE, "r") as f:
        samples = json.load(f)

    total_samples = len(samples)

    if total_samples == 0:
        print("Error: samples_train.json is empty. Cannot split.")
        return

    # Extract indices (all 558 entries)
    indices = list(range(total_samples))

    # Perform the split using scikit-learn's reproducible function
    # We use the list indices, not the sample data itself, to perform the split
    train_indices, val_indices = train_test_split(
        indices,
        test_size=VAL_SPLIT_PERCENTAGE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # 1. Update the original samples list with the correct 'split' field
    for i in indices:
        if i in train_indices:
            samples[i]["split"] = "train"
        elif i in val_indices:
            samples[i]["split"] = "val"
        else:
            # Should not happen, but safe check
            samples[i]["split"] = "unknown" 

    # 2. Write the updated list back to the file
    with open(TRAIN_JSON_FILE, "w") as f:
        json.dump(samples, f, indent=4)
        
    print(f"\n--- WUSU Validation Split Complete ---")
    print(f"Total Samples: {total_samples}")
    print(f"Assigned to 'train': {len(train_indices)}")
    print(f"Assigned to 'val':   {len(val_indices)}")
    print(f"File updated: {TRAIN_JSON_FILE.name} (Split is reproducible with random_state={RANDOM_STATE})")


if __name__ == "__main__":
    create_reproducible_val_split()