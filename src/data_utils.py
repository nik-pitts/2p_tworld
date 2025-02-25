import json


def save_human_data(player, filename="./data/human_play_data.json"):
    """🔹 Saves human gameplay data without overwriting existing data."""
    if not player.record or not player.human_play_data:
        print("No data to save.")
        return  # Do nothing if recording was off or no data exists

    try:
        # Fetch existing data
        with open(filename, "r") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []  # 파일이 없거나 손상된 경우 새로운 리스트 사용

    # Extend existing data with new data
    existing_data.extend(player.human_play_data)

    # Save updated data
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"📁 Human gameplay data saved to {filename}.")
