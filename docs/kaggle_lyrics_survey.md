# Kaggle lyrics survey for private scratch-model data

Reviewed: 2026-07-15

## Decision

Kaggle lyric datasets are useful for the private/personal-use training lane, but they should not be treated as clean/open-core sources by default.

The practical reason is simple: Kaggle can show a license for the dataset package, but that license may not prove the uploader had rights to redistribute the underlying lyric text. For this project, the right posture is:

- private/personal-use corpus: acceptable after local snapshot, hashes, schema inspection, and dedupe
- clean/open-core corpus: blocked unless item/source rights are proven
- public/commercial release: blocked by default

## Top candidates

1. `d3stron/english-music-lyrics-5-genres-500k`
   - Best first private-use target.
   - Good scale, English-only, genre labels.
   - Kaggle label observed as MIT, but lyric rights still need private-only treatment.
   - Local status: downloaded, normalized, and admitted to `scratch-private-extended-v1` as `kaggle_private_lyrics_english_5genres_500k`.
   - Local snapshot stats: 550,000 rows, 550,000 nonempty lyrics, about 175.3M rough tokens.

2. `nikhilnayak123/5-million-song-lyrics-dataset`
   - Best bulk-volume target.
   - License observed as Unknown; listing references a scraper.
   - Use only in private partition; download/sample before full ingest because it is large.
   - Local status: raw Kaggle snapshot downloaded and manifested; not yet normalized/admitted.
   - Local snapshot stats: 5,913,411 rows, 5,912,074 nonempty lyrics, about 2.10B rough tokens.

3. `carlosgdcj/genius-song-lyrics-with-language-information`
   - Useful because language labels help filtering.
   - Derived from the 5M lyrics dataset; inherited license/provenance risk.

4. `deepshah16/song-lyrics-dataset`
   - Kaggle label observed as CC0.
   - Still private-only unless underlying lyric rights are proven.

5. `evabot/spotify-lyrics-dataset`
   - Lower volume, but useful shape.
   - Listing indicates crawling from Spotify tracks and lyric web pages.

## Local status

Kaggle CLI/auth is configured locally through the repo virtual environment. The first two candidates have local snapshots:

- `d3stron/english-music-lyrics-5-genres-500k`: downloaded, normalized, admitted to the private profile, and verified through the source-governance commands.
- `nikhilnayak123/5-million-song-lyrics-dataset`: downloaded and raw-manifested; keep pending until a staged normalizer and dedupe pass are run.

To re-plan candidate priority, use:

```powershell
python scripts\plan_kaggle_lyrics_private_ingest.py --registry configs\datasets\kaggle_lyrics_candidate_registry.json
```

Then download the top candidate into:

```text
data/corpus_lake/raw/kaggle_private_lyrics/<kaggle_ref_slug>/<snapshot>
```

Preserve the original archive, file hashes, row counts, schema, and Kaggle metadata before converting anything into training records.

## Recommended ingest posture

Create a separate private source family:

```text
kaggle_private_lyrics
```

It should not enter `scratch-core-open-v1`. It can enter a private profile after:

- local archive snapshot saved
- license/provenance captured
- schema inspected
- row count measured
- duplicate overlap checked against `existing_song_lyrics_private_v1`
- explicit/profanity flags preserved instead of filtered

## Why this is still worth doing

For your actual goal — training a personal scratch lyric model — Kaggle is a real option. The clean/open-core work is slow because we are respecting rights evidence. The private Kaggle lane can be much faster and can give enough volume for 150M-300M experiments, as long as the artifacts are clearly labeled private-only.
