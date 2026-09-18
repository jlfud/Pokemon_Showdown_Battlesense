"""
Download and filter Gen 9 OU battles from Hugging Face dataset.

This script:
1. Streams the dataset (doesn't download all 38GB at once)
2. Filters for Gen 9 OU format and rating 1200-1800
3. Samples ~50k battles
4. Saves to JSON Lines format for easy parsing
"""

from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm


def download_gen9_ou_data(
    target_count=50000,
    min_rating=1200,
    max_rating=1800,
    output_dir="data/raw",
    format_id="gen8ou"  # Changed to gen8ou (more data available)
):
    """
    Download and filter Gen 8 OU battles.

    Args:
        target_count: Number of battles to download (default 50k)
        min_rating: Minimum player rating (default 1200)
        max_rating: Maximum player rating (default 1800)
        output_dir: Where to save the data
        format_id: Format to download (default gen8ou)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{format_id}_battles.jsonl"

    print(f"Loading dataset from Hugging Face...")
    print(f"Target: {target_count} {format_id.upper()} battles")
    print(f"Rating filter: {min_rating}-{max_rating}")

    # Load dataset in streaming mode
    ds = load_dataset(
        "milkkarten/pokemon-showdown-replays-merged",
        split="train",
        streaming=True
    )

    # Filter for specified format with rating in range
    def filter_fn(battle):
        # Use formatid field (e.g., "gen8ou") instead of format (e.g., "[Gen 8] OU")
        is_format_match = battle.get("formatid") == format_id
        rating = battle.get("rating")
        has_valid_rating = rating is not None and min_rating <= rating <= max_rating
        return is_format_match and has_valid_rating

    filtered_ds = ds.filter(filter_fn)

    battles_saved = 0

    print("\nDownloading battles...")
    with open(output_file, 'w') as f:
        pbar = tqdm(total=target_count, desc="Saving battles")

        for battle in filtered_ds:
            # Save battle as JSON line
            f.write(json.dumps(battle) + '\n')
            battles_saved += 1
            pbar.update(1)

            # Stop when we hit target
            if battles_saved >= target_count:
                break

        pbar.close()

    print(f"\n✓ Downloaded {battles_saved} battles")
    print(f"  Saved to: {output_file}")

    # Save metadata
    metadata = {
        'total_battles': battles_saved,
        'format_id': format_id,
        'min_rating': min_rating,
        'max_rating': max_rating
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata saved to: {metadata_file}")

    return output_file


def preview_data(output_file="data/raw/gen8ou_battles.jsonl", num_samples=3):
    """Preview downloaded battles."""
    print(f"\nPreviewing {num_samples} battles from {output_file}:\n")

    with open(output_file) as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            battle = json.loads(line)
            print(f"Battle {i+1}:")
            print(f"  ID: {battle.get('id')}")
            print(f"  Format: {battle.get('format')}")
            print(f"  Players: {', '.join(battle.get('players', []))}")
            print(f"  Rating: {battle.get('rating')}")
            print(f"  Turns: {battle.get('turns')}")
            print(f"  Log length: {len(battle.get('log', ''))} characters")
            print()


if __name__ == "__main__":
    # Download data
    output_file = download_gen9_ou_data(
        target_count=50000,  # Start with 50k battles
        min_rating=1200,
        max_rating=1800,
        format_id="gen8ou"  # Using Gen 8 OU (more data available)
    )

    # Preview the data
    preview_data(output_file)
