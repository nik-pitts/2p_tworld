import json
import os
import time
import traceback


def save_human_data(player, filename="./data/human_play_data.json"):
    """🔹 Saves human gameplay data by appending to existing data."""
    if not player.record or not player.human_play_data:
        print("No data to save.")
        return  # Do nothing if recording was off or no data exists

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Initialize with empty list (in case file doesn't exist)
    existing_data = []

    # Try to read existing data
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                existing_data = json.load(f)
                print(f"Read {len(existing_data)} existing entries from {filename}")

            if not isinstance(existing_data, list):
                print(
                    f"Warning: Existing file {filename} does not contain a list. Creating backup and starting new file."
                )
                os.rename(filename, filename + ".backup")
                existing_data = []

        except json.JSONDecodeError:
            print(
                f"Warning: Existing file {filename} contains invalid JSON. Creating backup and starting new file."
            )
            os.rename(filename, filename + ".backup")
        except Exception as e:
            print(f"Error reading existing file {filename}: {str(e)}")
            print(traceback.format_exc())

    # Add new data
    new_entries = len(player.human_play_data)
    existing_data.extend(player.human_play_data)

    # Save updated data
    try:
        with open(filename, "w") as f:
            json.dump(existing_data, f, indent=4)
        print(
            f"📁 Human gameplay data saved to {filename}. Added {new_entries} new entries for a total of {len(existing_data)}."
        )
    except Exception as e:
        print(f"Error saving data to {filename}: {str(e)}")
        print(traceback.format_exc())

        # Try to save to a backup file
        backup_filename = filename + f".backup-{int(time.time())}"
        try:
            with open(backup_filename, "w") as f:
                json.dump(existing_data, f, indent=4)
            print(f"📁 Data saved to backup file {backup_filename} instead.")
        except Exception as e:
            print("Error saving to backup file as well. Data may be lost.")
