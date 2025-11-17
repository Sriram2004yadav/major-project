#!/bin/bash

# --- Configuration ---
SOURCE_DIR="TSCD"
DEST_DIR="TSCD_processed"
# --- End Configuration ---

echo "Starting TSCD dataset pre-processing..."

# 1. Create the new directory structure
echo "Creating new directories in $DEST_DIR..."
mkdir -p "$DEST_DIR/train/imgs"
mkdir -p "$DEST_DIR/train/change_labels_adj"
mkdir -p "$DEST_DIR/val/imgs"
mkdir -p "$DEST_DIR/val/change_labels_adj"
mkdir -p "$DEST_DIR/test/imgs"
mkdir -p "$DEST_DIR/test/change_labels_adj"

# 2. Define the splits to process
SPLITS=("train" "val" "test")

for split in "${SPLITS[@]}"; do
  echo "--- Processing split: $split ---"
  
  # Find all the base image files (e.g., 0.jpg, 1.jpg) in the first folder
  SOURCE_A_DIR="$SOURCE_DIR/2016_2018/$split/A"
  
  if [ ! -d "$SOURCE_A_DIR" ]; then
    echo "Warning: Source directory $SOURCE_A_DIR not found. Skipping split '$split'."
    continue
  fi
  
  # Loop through every .jpg file in the reference directory
  for file_path in "$SOURCE_A_DIR"/*.jpg; do
  
    # Get just the filename, e.g., "0.jpg"
    filename=$(basename "$file_path")
    # Get just the index, e.g., "0"
    index="${filename%.*}"

    echo "Copying and renaming index: $index"

    # --- Define all source and destination paths ---
    
    # Handle the inconsistent 'label' vs 'OUT' folder for 2018-2020
    label_dir_2018_2020="OUT" # Default for train/val
    if [ "$split" == "test" ]; then
      label_dir_2018_2020="label" # For test
    fi

    # Source Paths (Your raw data)
    src_img_2016="TSCD/2016_2018/$split/A/$filename"
    src_img_2018="TSCD/2016_2018/$split/B/$filename"
    src_img_2020="TSCD/2018_2020/$split/B/$filename"
    src_img_2022="TSCD/2020_2022/$split/B/$filename"
    
    src_lbl_16_18="TSCD/2016_2018/$split/label/$filename"
    src_lbl_18_20="TSCD/2018_2020/$split/$label_dir_2018_2020/$filename"
    src_lbl_20_22="TSCD/2020_2022/$split/OUT/$filename"

    # Destination Paths (The new structure)
    dest_img_2016="$DEST_DIR/$split/imgs/${index}_2016.jpg"
    dest_img_2018="$DEST_DIR/$split/imgs/${index}_2018.jpg"
    dest_img_2020="$DEST_DIR/$split/imgs/${index}_2020.jpg"
    dest_img_2022="$DEST_DIR/$split/imgs/${index}_2022.jpg"
    
    dest_lbl_16_18="$DEST_DIR/$split/change_labels_adj/${index}_2016-2018.jpg"
    dest_lbl_18_20="$DEST_DIR/$split/change_labels_adj/${index}_2018-2020.jpg"
    dest_lbl_20_22="$DEST_DIR/$split/change_labels_adj/${index}_2020-2022.jpg"

    # --- Run the 7 copy commands ---
    cp "$src_img_2016" "$dest_img_2016"
    cp "$src_img_2018" "$dest_img_2018"
    cp "$src_img_2020" "$dest_img_2020"
    cp "$src_img_2022" "$dest_img_2022"
    
    cp "$src_lbl_16_18" "$dest_lbl_16_18"
    cp "$src_lbl_18_20" "$dest_lbl_18_20"
    cp "$src_lbl_20_22" "$dest_lbl_20_22"
    
  done
done

echo "--- File copying complete! ---"

# --- Step 3: Create the JSON files ---

echo "Creating JSON sample files..."

# For train
echo "[" > "$DEST_DIR/samples_train.json"
for f in "$DEST_DIR/train/imgs/"*_2016.jpg; do
  [ -e "$f" ] || continue # Handle case where no files are found
  filename=$(basename "$f")
  index="${filename%_2016.jpg}"
  echo "  {\"index\": \"$index\"}," >> "$DEST_DIR/samples_train.json"
done
# Remove trailing comma and close bracket
sed -i '$ s/,$//' "$DEST_DIR/samples_train.json"
echo "]" >> "$DEST_DIR/samples_train.json"

# For val
echo "[" > "$DEST_DIR/samples_val.json"
for f in "$DEST_DIR/val/imgs/"*_2016.jpg; do
  [ -e "$f" ] || continue
  filename=$(basename "$f")
  index="${filename%_2016.jpg}"
  echo "  {\"index\": \"$index\"}," >> "$DEST_DIR/samples_val.json"
done
sed -i '$ s/,$//' "$DEST_DIR/samples_val.json"
echo "]" >> "$DEST_DIR/samples_val.json"

# For test
echo "[" > "$DEST_DIR/samples_test.json"
for f in "$DEST_DIR/test/imgs/"*_2016.jpg; do
  [ -e "$f" ] || continue
  filename=$(basename "$f")
  index="${filename%_2016.jpg}"
  echo "  {\"index\": \"$index\"}," >> "$DEST_DIR/samples_test.json"
done
sed -i '$ s/,$//' "$DEST_DIR/samples_test.json"
echo "]" >> "$DEST_DIR/samples_test.json"

echo "--- All Done! Your 'TSCD_processed' folder is ready. ---"