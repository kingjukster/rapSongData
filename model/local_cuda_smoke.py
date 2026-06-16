"""Print local CUDA and PyTorch GPU information."""

from __future__ import annotations

import json


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is not installed. Run: pip install -r requirements-local-cuda.txt") from exc

    result = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "capability": f"{props.major}.{props.minor}",
                "total_vram_gb": round(props.total_memory / 1024**3, 2),
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
