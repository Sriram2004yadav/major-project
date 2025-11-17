
rt json
from pathlib import Path
import random

# --- CONFIG ---
DATA_ROOT = Path("./WUSU_processed")
TRAIN_JSON_FILE = DATA_ROOT / "samples_train.json"
# The original paper used approximately a 90% train / 10% val split on the available samples.
VAL_SPLIT_PERCENTAGE = 0.10 
# --- CONFIG ---

def create_val_split():
    if not TRAIN_JSON_FILE.exists():
        print(f"Error: {TRAIN_JSON_FILE} not found. Did you run create_wusu_jsons.py first?")
        return

    with open(TRAIN_JSON_FILE, "r") as f:
        samples = json.load(f)

    total_samples = len(samples)
    val_count = int(total_samples * VAL_SPLIT_PERCENTAGE)
    train_count = total_samples - val_count

    # 1. Randomly select the validation indices
    # We randomize the list first to avoid always picking the last 56 tiles, 
    # ensuring the validation set is geographically dispersed.
    random.seed(42) # Set seed for reproducibility
    random.shuffle(samples) 

    # 2. Assign the last 'val_count' samples to 'val'
    for i, sample in enumerate(samples):
        if i < train_count:
            # Assign the first 90% to 'train'
            samples[i]["split"] = "train"
        else:
            # Assign the remaining 10% to 'val'
            samples[i]["split"] = "val"

    # 3. Write the updated list back to the file
    with open(TRAIN_JSON_FILE, "w") as f:
        json.dump(samples, f, indent=4)
        
    print(f"\n--- WUSU Validation Split Complete ---")
    print(f"Total Samples Processed: {total_samples}")
    print(f"Assigned to 'train': {train_count}")
    print(f"Assigned to 'val':   {val_count} (10% of total)")
    print(f"File updated: {TRAIN_JSON_FILE.name}")


if __name__ == "__main__":
    create_val_split()
