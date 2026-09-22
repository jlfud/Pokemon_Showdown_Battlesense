# Local dataset

The Phase 1 dataset is generated locally and intentionally excluded from Git.

## Raw data

- File: `data/raw/gen8ou_battles.jsonl`
- Battles: 10,000 unique replay IDs
- Format: `gen8ou`
- Dataset rating field: 1200 through 1800 inclusive
- Size: 102,721,720 bytes
- SHA-256: `66485a804cf058f9fb2174f04caafda2a1c219d8ea5358e4250501b2759191c8`

Regenerate or resume it with:

```bash
python -m src.download_data_v2 --target-count 10000
```

The downloader uses Parquet predicate pushdown and de-duplicates replay IDs when
resuming. Do not force-add the JSONL file to Git.

## Processed data

Run:

```bash
python -m src.data.prepare_dataset
```

This produces deterministic battle-level splits under `data/processed/`:

| Split | Battles | Action examples |
| --- | ---: | ---: |
| Train | 6,958 | 343,334 |
| Validation | 1,477 | 74,095 |
| Test | 1,565 | 76,830 |

There is no battle-ID overlap between splits. Each example represents the state
at the start of a turn and one player's voluntary move or switch. Forced
replacements and switches caused by pivot moves are not labels. Phase 2 examples
also include `team_species`, `opponent_team_species`, and `available_switches`;
the last field excludes the active and fainted Pokemon and is used to mask
illegal switch predictions.
