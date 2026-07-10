# Legacy model paths

Python files in this directory are compatibility wrappers for the package under
`src/rap_song_data/`. Existing commands such as
`python model/train_local_cuda.py` continue to work, but new workflows should
use the console commands documented in the root `README.md`.

Historical generated datasets and model artifacts remain here so existing run
manifests and adapter paths do not break. New source code and configuration do
not belong in this directory.
