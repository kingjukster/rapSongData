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
   - Local status: raw Kaggle snapshot downloaded and manifested; first rap-only pilot normalized, exact-deduped, and admitted to `scratch-private-extended-v1` as `kaggle_private_lyrics_5m_genius_scrape`.
   - Local snapshot stats: 5,913,411 rows, 5,912,074 nonempty lyrics, about 2.10B rough tokens.
   - Pilot stats: scanned 1,864,369 rows to approve 250,000 rap records, rejected 78,858 exact duplicates against the existing scratch corpus, and added about 165.3M rough private-only tokens.

3. `carlosgdcj/genius-song-lyrics-with-language-information`
   - Useful because language labels help filtering.
   - Derived from the 5M lyrics dataset; inherited license/provenance risk.
   - Local status: downloaded and raw-manifested.
   - Local snapshot stats: 5,134,856 rows, 5,134,856 nonempty lyrics, about 2.07B rough tokens; includes language labels.

4. `deepshah16/song-lyrics-dataset`
   - Kaggle label observed as CC0.
   - Still private-only unless underlying lyric rights are proven.
   - Local status: downloaded and raw-manifested.
   - Local snapshot stats: 6,027 CSV rows, 5,981 nonempty lyrics, about 2.8M rough tokens; JSON sidecars preserved.

5. `evabot/spotify-lyrics-dataset`
   - Lower volume, but useful shape.
   - Listing indicates crawling from Spotify tracks and lyric web pages.
   - Local status: downloaded and raw-manifested.
   - Local snapshot stats: 8,674 rows, 8,674 nonempty lyrics, about 3.6M rough tokens.

## Local status

Kaggle CLI/auth is configured locally through the repo virtual environment. Current local snapshots:

- `d3stron/english-music-lyrics-5-genres-500k`: downloaded, normalized, admitted to the private profile, and verified through the source-governance commands.
- `nikhilnayak123/5-million-song-lyrics-dataset`: downloaded and raw-manifested; a bounded rap250k pilot is normalized, exact-deduped against `data/scratch/v1/dedupe.sqlite3`, admitted to the private profile, and verified through the source-governance commands.
- `carlosgdcj/genius-song-lyrics-with-language-information`: downloaded and raw-manifested; language labels make it a good staged English-only normalizer target.
- `neisse/scrapped-lyrics-from-6-genres`: downloaded and raw-manifested; 379,931 nonempty rows and about 100.3M rough tokens.
- `devdope/900k-spotify`: downloaded and raw-manifested; primary CSV has 551,443 nonempty lyric rows and about 242.7M rough tokens; license label is noncommercial.
- `bwandowando/spotify-songs-with-attributes-and-lyrics`: downloaded and raw-manifested; song-level lyric file has about 955k nonempty rows, plus a timestamp/alignment file with line-level rows.
- Smaller raw-manifested sets: `deepshah16/song-lyrics-dataset`, `evabot/spotify-lyrics-dataset`, `notshrirang/spotify-million-song-dataset`, `suraj520/music-dataset-song-information-and-lyrics`, `paultimothymooney/poetry`, and `juicobowley/drake-lyrics`.

## Newly pulled raw snapshots

| Kaggle ref | Local source id | Rows or docs | Nonempty lyric rows/docs | Rough tokens | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `carlosgdcj/genius-song-lyrics-with-language-information` | `kaggle_private_lyrics_genius_language_info` | 5,134,856 | 5,134,856 | 2,065.1M | raw-manifested |
| `neisse/scrapped-lyrics-from-6-genres` | `kaggle_private_lyrics_neisse_6_genres` | 379,931 | 379,931 | 100.3M | raw-manifested |
| `devdope/900k-spotify` | `kaggle_private_lyrics_devdope_900k_spotify` | 551,443 | 551,443 | 242.7M | raw-manifested |
| `bwandowando/spotify-songs-with-attributes-and-lyrics` | `kaggle_private_lyrics_bwandowando_spotify_960k` | 37,474,412 total rows across song and timestamp files | 955,307 song-level lyrics | 332.1M | raw-manifested |
| `notshrirang/spotify-million-song-dataset` | `kaggle_private_lyrics_spotify_million_song` | 57,650 | 57,650 | 17.6M | raw-manifested |
| `evabot/spotify-lyrics-dataset` | `kaggle_private_lyrics_spotify_10k` | 8,674 | 8,674 | 3.6M | raw-manifested |
| `deepshah16/song-lyrics-dataset` | `kaggle_private_lyrics_deepshah_artist_collection` | 6,027 | 5,981 | 2.8M | raw-manifested |
| `suraj520/music-dataset-song-information-and-lyrics` | `kaggle_private_lyrics_suraj_music_info` | 799 | 799 | 1.4M | raw-manifested |
| `paultimothymooney/poetry` | `kaggle_private_lyrics_paultimothy_song_lyrics_txt` | 49 text files | 49 text files | 1.7M | raw-manifested |
| `juicobowley/drake-lyrics` | `kaggle_private_lyrics_drake_lyrics` | 290 | 288 | 0.2M | raw-manifested |

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
