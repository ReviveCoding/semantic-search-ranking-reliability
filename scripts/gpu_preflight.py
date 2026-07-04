from __future__ import annotations

import json
import platform
import sys

import numpy as np
import torch


def main() -> None:
    payload: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }

    if not torch.cuda.is_available():
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit("GPU_PREFLIGHT=FAIL")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)

    left = torch.randn(
        (512, 512),
        device=device,
        dtype=torch.float16,
    )

    right = torch.randn(
        (512, 512),
        device=device,
        dtype=torch.float16,
    )

    result = left @ right
    torch.cuda.synchronize()

    payload.update(
        {
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_count": torch.cuda.device_count(),
            "gpu_capability": list(
                torch.cuda.get_device_capability(device)
            ),
            "gpu_total_memory_gb": round(
                properties.total_memory / 1024**3,
                2,
            ),
            "matmul_checksum": round(
                float(result.float().mean().item()),
                6,
            ),
        }
    )

    print(json.dumps(payload, indent=2, sort_keys=True))
    print("GPU_PREFLIGHT=PASS")


if __name__ == "__main__":
    main()