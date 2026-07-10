# rap-song-data

Tools for curating a rap-lyrics corpus, building supervised and preference
datasets, fine-tuning lyric-generation models, and evaluating generated verses.

## Project layout

```text
src/rap_song_data/   importable Python package
configs/             pipeline, dataset, training, evaluation, and prompt configs
notebooks/           exploratory notebooks
scripts/             experiment launchers and operational utilities
tests/               automated tests
docs/                maintained guides and experiment documentation
data/                source and derived datasets
reports/             cross-run reports and audits
runs/                complete experiment records
model/               legacy wrappers and historical local model artifacts
```

Historical data, run, and evidence trees intentionally keep their existing
paths because their manifests contain provenance links to those locations.

## Setup

Create an environment and install the project in editable mode:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[data,dev]"
```

For local GPU training, install a CUDA-compatible PyTorch build first, then:

```powershell
python -m pip install -e ".[data,cuda,dev]"
```

The existing dependency list remains available in
`requirements-local-cuda.txt` for environments that do not use extras.

## Common commands

```powershell
rap-pipeline --help
rap-clean-corpus --help
rap-build-training-data --config configs/datasets/dataset_config.json
rap-train-local --config configs/training/local_cuda_config.example.json
rap-generate-local --help
python -m pytest
```

The old root and `model/` file commands remain as compatibility wrappers, so
existing automation can be migrated gradually. New code should import from
`rap_song_data` or use the installed console commands.

See [the fast-pipeline guide](docs/guides/fast_pipeline.md) and
[the modeling guide](docs/guides/modeling.md) for end-to-end workflows.
