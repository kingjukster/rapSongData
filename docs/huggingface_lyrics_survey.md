# Hugging Face lyrics survey for private scratch-model data

Reviewed: 2026-07-15

## Decision

Hugging Face is a strong acquisition source because many datasets are already in CSV, JSON, or Parquet and can be loaded with ML tooling. Treat lyric text as private/personal-use unless underlying lyric rights are separately verified.

## Local status

Public Hugging Face pulls are stored under:

```text
data/corpus_lake/raw/huggingface_lyrics/<namespace>__<repo>/20260715_hf_public_snapshot
```

The first public batches downloaded and raw-manifested 15 datasets/artifact sets:

| Hugging Face dataset | Local source id | Rows/docs | Nonempty lyrics/docs | Rough tokens | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `theelderemo/genius-lyrics-cleaned` | `hf_private_lyrics_theelderemo__genius_lyrics_cleaned` | 3,179,588 | 3,179,588 | 1,160.9M | raw-manifested |
| `PJMixers-Dev/bigdata-pw_Lyrics1M-en` | `hf_private_lyrics_pjmixers_dev__bigdata_pw_lyrics1m_en` | 553,131 | 553,131 | 198.1M | raw-manifested |
| `HowitzerDeBoullion/Multi-Lingual-Lyrics-for-Genre-Classification` | `hf_private_lyrics_howitzerdeboullion__multi_lingual_lyrics_for_genre_classification` | 596,236 | 596,201 | 177.3M | raw-manifested |
| `Cropinky/rap_lyrics_english` | `hf_private_lyrics_cropinky__rap_lyrics_english` | 47 text files | 46 text files | 11.0M | raw-manifested |
| `halaction/song-lyrics` | `hf_private_lyrics_halaction__song_lyrics` | 53,876 | 53,876 | 19.4M | raw-manifested |
| `Koyd111/hiphop-song-lyrics` | `hf_private_lyrics_koyd111__hiphop_song_lyrics` | 900 | 900 | 2.6M | raw-manifested |
| `nateraw/rap-lyrics-v1` | `hf_private_lyrics_nateraw__rap_lyrics_v1` | 2,350 | 2,350 | 1.9M | raw-manifested |
| `nateraw/rap-lyrics-v2` | `hf_private_lyrics_nateraw__rap_lyrics_v2` | 7,319 | 7,319 | 0.7M | raw-manifested |
| `vancenceho/spotify-lyrics` | `hf_private_lyrics_vancenceho__spotify_lyrics` | 57,650 | 57,650 | 17.6M | raw-manifested |
| `vancenceho/spotify-lyrics-clean` | `hf_private_lyrics_vancenceho__spotify_lyrics_clean` | 57,438 | 57,438 | 16.3M | raw-manifested |
| `theelderemo/lyrics-database` | `hf_private_lyrics_theelderemo__lyrics_database` | 49,985 | 49,985 | 15.9M | raw-manifested |
| `smgriffin/modern-pop-lyrics` | `hf_private_lyrics_smgriffin__modern_pop_lyrics` | 17,174 | 17,174 | 6.6M | raw-manifested |
| `theodoredc/hiphop-lyrics` | `hf_private_lyrics_theodoredc__hiphop_lyrics` | 15,124 | 15,124 | 10.9M | raw-manifested |
| `amishshah/song_lyrics` | `hf_private_lyrics_amishshah__song_lyrics` | 3,374,198 | 3,374,198 | 1,378.7M | raw-manifested |
| `asigalov61/Lyrics-MIDI-Dataset` | `hf_private_lyrics_asigalov61__lyrics_midi_dataset` | selected binary/zip artifacts | not counted yet | not counted yet | raw-manifested selected artifacts |

Batch total before dedupe, excluding unextracted Asigalov binary/zip artifacts:

- 7,965,016 rows/docs
- 7,964,980 nonempty lyrics/docs
- about 3.02B rough tokens

## Deferred large candidates

- `Dr3dre/Genius-song-lyrics-cleaned`: about 18.4 GB of Parquet; likely overlaps the cleaned Genius/Kaggle sources.
- `amishshah/song_lyrics`: pulled and raw-manifested with D:-pinned cache, then duplicate cache removed.
- `asigalov61/Lyrics-MIDI-Dataset`: selected lyric/corpus artifacts pulled; model files skipped.

## Recommended next step

Normalize and exact-dedupe `theelderemo/genius-lyrics-cleaned` first, then decide whether the larger `Dr3dre` mirror is still worth downloading.
