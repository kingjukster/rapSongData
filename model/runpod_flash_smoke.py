"""Run a small Runpod Flash GPU smoke test.

This is the first thing to run after `pip install runpod-flash` and
`flash login`. It provisions a short-lived GPU endpoint and returns basic CUDA
information to your local terminal.
"""

from __future__ import annotations

import asyncio

try:
    from runpod_flash import Endpoint, GpuType
except ImportError as exc:  # pragma: no cover - only hit before setup
    raise SystemExit("runpod-flash is not installed. Run: pip install runpod-flash") from exc


@Endpoint(
    name="rap-lyrics-gpu-smoke-test",
    gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
    workers=(0, 1),
    dependencies=["torch"],
    execution_timeout_ms=300_000,
)
async def gpu_smoke_test() -> dict:
    """Return basic GPU and CUDA information from the remote worker."""
    import torch

    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


async def main() -> None:
    try:
        result = await gpu_smoke_test()
    except RuntimeError as exc:
        message = str(exc)
        if "not found" in message and "deploy it first" in message:
            raise SystemExit(
                "Runpod Flash endpoint is not deployed yet.\n\n"
                "Run this once from the project root:\n"
                "  flash deploy --python-version 3.12\n\n"
                "Then retry:\n"
                "  python model/runpod_flash_smoke.py\n\n"
                "The root .gitignore excludes local datasets so Flash does not upload the multi-GB corpus."
            ) from exc
        raise
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
