# Configuration

- `pipeline.yaml`: primary curation and generation defaults.
- `datasets/`: dataset builders and architecture seed definitions.
- `training/`: local CUDA, QLoRA, and Runpod training profiles.
- `evaluation/`: judge and quality-ranker profiles.
- `prompts/`: versioned prompt inputs used by benchmarks.

Paths inside configuration files are resolved from the repository working
directory. Run commands from the project root unless a command says otherwise.
