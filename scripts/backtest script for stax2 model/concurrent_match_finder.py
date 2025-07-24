#!/usr/bin/env python3
"""
Stax - Concurrent Match Finder (with Data Length)

This script scans a directory of match data files, identifies games
that start at the same time, and prints the number of data entries
in each file to help diagnose data completeness issues.

The script assumes filenames are in the format:
..._YYYY-MM-DD_HH-mm-ss.json
"""

import os
import json
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
# This script should be in the /scripts folder, so we navigate up and then down to /data
ROOT_DIR = Path(__file__).resolve().parent.parent
# We will look for concurrent games in the Backtest data
DATA_DIR = ROOT_DIR / "data" / "Backtest"

def find_concurrent_games(directory: Path):
    """
    Scans the given directory for JSON files, groups them by start time,
    and prints the length of each file.

    Args:
        directory (Path): The path to the directory containing match data files.
    """
    print(f"Scanning for concurrent matches in: {directory}\n")

    if not directory.exists():
        print(f"Error: Directory not found at {directory}")
        return

    # Use a defaultdict to easily group files by a key
    games_by_time = defaultdict(list)

    # Get all json files from the directory
    match_files = list(directory.glob('*.json'))

    # Loop through all found files
    for file_path in match_files:
        filename_stem = file_path.stem
        
        try:
            # The key is the date and time, e.g., "2025-04-12_17-28-02"
            parts = filename_stem.split('_')
            time_key = f"{parts[-2]}_{parts[-1]}"

            # Open the file and get the number of entries
            with open(file_path, 'r') as f:
                data = json.load(f)
                num_entries = len(data)

            # Store the filename and its length
            games_by_time[time_key].append((file_path.name, num_entries))

        except (IndexError, json.JSONDecodeError) as e:
            print(f"Could not parse or read file: {file_path.name} - Error: {e}")

    # Now, find the groups with more than one game
    concurrent_groups = {
        time: files for time, files in games_by_time.items() if len(files) > 1
    }

    if not concurrent_groups:
        print("No concurrent games found in the directory.")
        return

    print(f"Found {len(concurrent_groups)} time slots with concurrent games:\n")

    # Sort the groups by time for a clean output
    sorted_times = sorted(concurrent_groups.keys())

    for time_key in sorted_times:
        files_with_counts = concurrent_groups[time_key]
        print(f"--- Games starting at {time_key} ({len(files_with_counts)} matches) ---")
        
        # Sort files alphabetically within the group
        sorted_files = sorted(files_with_counts, key=lambda x: x[0])
        
        for file_name, count in sorted_files:
            # Add a visual warning for files with suspiciously few entries
            warning_flag = "⚠️  <-- WARNING: Short file" if count < 100 else ""
            print(f"  - {file_name:<75} | Entries: {count:<5} {warning_flag}")
        print("") # Add a newline for readability

if __name__ == '__main__':
    find_concurrent_games(DATA_DIR)
