from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_generation_ckpt() -> Path:
    return (
        repo_root()
        / "omni_ckpts"
        / "hf_mobile_ov"
        / "stage1_joint_openvid_fullmobile_o_fulldit_diffonly_initlatest_bs64_v2_20260429_8gpu_60k.pt"
    )


def default_smolvlm2_ckpt() -> Path:
    return repo_root() / "omni_ckpts" / "smolvlm2_500m" / "smolvlm2_500m.pt"


def resolve_path(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def build_backend_env() -> dict:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    existing = env.get("PYTHONPATH", "")
    root = repo_root()
    env["PYTHONPATH"] = str(root) if not existing else f"{root}:{existing}"
    return env


def run_backend_python(
    script_relpath: str,
    argv: Iterable[str],
) -> int:
    root = repo_root()
    script_path = root / script_relpath
    if not script_path.exists():
        raise FileNotFoundError(f"Backend script not found: {script_path}")

    command: List[str] = [sys.executable, str(script_path), *list(argv)]
    print("Running backend command:")
    print(shlex.join(command))
    completed = subprocess.run(
        command,
        cwd=str(root),
        env=build_backend_env(),
        check=False,
    )
    return int(completed.returncode)
