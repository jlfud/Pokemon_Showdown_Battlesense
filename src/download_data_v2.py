"""
Download Gen 8 OU battles - Improved version with resumable downloads.

This version:
1. Downloads in batches with checkpoints
2. Can resume from where it left off
3. Uses append mode to accumulate data
4. Has better error handling
"""

from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm
import time


def download_battles_resumable(
    target_count=50000,
    min_rating=1200,
    max_rating=1800,
    format_id="gen8ou",
    output_dir="data/raw",
    batch_size=1000,
    max_process=200000  # Stop after checking this many battles if target not reached
):
    """
    Download battles with resumable checkpoints.

    Args:
        target_count: Number of battles to download
        min_rating: Minimum player rating
        max_rating: Maximum player rating
        format_id: Format to download (gen8ou, gen9ou, etc.)
        output_dir: Where to save the data
        batch_size: Save progress every N battles
        max_process: Maximum battles to process before giving up
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{format_id}_battles.jsonl"

    # Check if we have existing data
    existing_count = 0
    if output_file.exists():
        with open(output_file) as f:
            existing_count = sum(1 for _ in f)
        print(f"Found {existing_count} existing battles, resuming download...")

        if existing_count >= target_count:
            print(f"Already have {existing_count} battles (target: {target_count})")
            return output_file
    else:
        # Create new file
        output_file.touch()

    remaining = target_count - existing_count

    print(f"\nDownloading {remaining} more {format_id.upper()} battles")
    print(f"Rating filter: {min_rating}-{max_rating}")
    print(f"Will check up to {max_process:,} battles\n")

    # Load dataset in streaming mode
    print("Loading HuggingFace dataset...")
    ds = load_dataset(
        "milkkarten/pokemon-showdown-replays-merged",
        split="train",
        streaming=True
    )

    battles_saved = 0
    battles_processed = 0

    # Open file in append mode
    with open(output_file, 'a') as f:
        pbar = tqdm(total=remaining, desc="Saving battles")

        try:
            for battle in ds:
                battles_processed += 1

                # Check format and rating
                if battle.get("formatid") == format_id:
                    rating = battle.get("rating")
                    if rating is not None and min_rating <= rating <= max_rating:
                        # Save battle
                        f.write(json.dumps(battle) + '\n')
                        f.flush()  # Ensure data is written
                        battles_saved += 1
                        pbar.update(1)

                        # Check if we've hit target
                        if battles_saved >= remaining:
                            break

                # Update progress bar description with stats
                if battles_processed % 1000 == 0:
                    hit_rate = (battles_saved / battles_processed * 100) if battles_processed > 0 else 0
                    pbar.set_description(f"Saving battles (checked {battles_processed:,}, hit rate {hit_rate:.2f}%)")

                # Safety check: don't process forever
                if battles_processed >= max_process:
                    print(f"\n⚠ Reached max process limit ({max_process:,} battles)")
                    print(f"   Only found {battles_saved} matching battles")
                    break

        except KeyboardInterrupt:
            print("\n\n⚠ Download interrupted by user")
        except Exception as e:
            print(f"\n\n⚠ Error during download: {e}")
        finally:
            pbar.close()

    total_saved = existing_count + battles_saved

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Total battles saved: {total_saved:,}")
    print(f"  New battles in this run: {battles_saved:,}")
    print(f"  Battles processed: {battles_processed:,}")
    if battles_processed > 0:
        print(f"  Hit rate: {(battles_saved / battles_processed * 100):.2f}%")
    print(f"  Saved to: {output_file}")
    print(f"{'='*60}\n")

    # Save metadata
    metadata = {
        'total_battles': total_saved,
        'format_id': format_id,
        'min_rating': min_rating,
        'max_rating': max_rating,
        'battles_processed': battles_processed
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_file


def preview_data(output_file="data/raw/gen8ou_battles.jsonl", num_samples=3):
    """Preview downloaded battles."""
    if not Path(output_file).exists():
        print(f"File not found: {output_file}")
        return

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
    # Download data (resumable) - NO LIMIT, scan entire dataset
    output_file = download_battles_resumable(
        target_count=50000,
        min_rating=1200,
        max_rating=1800,
        format_id="gen8ou",
        batch_size=1000,
        max_process=30000000  # Scan up to 30M battles (entire dataset)
    )

    # Preview the data
    preview_data(output_file)
