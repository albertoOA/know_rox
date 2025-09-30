#!/bin/bash

# Get script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Target local folder
TARGET_DIR="$SCRIPT_DIR/../../csv/explanatory_narratives_cra/acxon_based"
mkdir -p "$TARGET_DIR"

# GitHub config
USER="albertoOA"
REPO="explanatory_narratives_cra"
BRANCH="rox_dev"
FOLDER_PATH="csv/acxon_based/know_rox/narratives/unknown"

# Get all files from the folder in the branch
curl -s "https://api.github.com/repos/$USER/$REPO/contents/$FOLDER_PATH?ref=$BRANCH" |
grep '"download_url"' |
cut -d '"' -f 4 |
while read -r file_url; do
    file_name=$(basename "$file_url")
    echo "Downloading $file_name..."
    curl -s -L "$file_url" -o "$TARGET_DIR/$file_name"
done

echo "✅ All files from $FOLDER_PATH in branch $BRANCH downloaded to $TARGET_DIR"

FOLDER_PATH="csv/acxon_based/know_rox/narratives/ssd_case_inspection"
# Get all files from the folder in the branch
curl -s "https://api.github.com/repos/$USER/$REPO/contents/$FOLDER_PATH?ref=$BRANCH" |
grep '"download_url"' |
cut -d '"' -f 4 |
while read -r file_url; do
    file_name=$(basename "$file_url")
    echo "Downloading $file_name..."
    curl -s -L "$file_url" -o "$TARGET_DIR/$file_name"
done

echo "✅ All files from $FOLDER_PATH in branch $BRANCH downloaded to $TARGET_DIR"