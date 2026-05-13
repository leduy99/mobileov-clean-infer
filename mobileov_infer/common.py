from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_backend_repo() -> Path:
    return Path(__file__).resolve().parents[2] / "Omni-Video-smolvlm2"


def resolve_backend_repo(explicit: Optional[str] = None) -> Path:
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        Path(os.environ["MOBILEOV_BACKEND_REPO"]).expanduser() if os.environ.get("MOBILEOV_BACKEND_REPO") else None,
        _default_backend_repo(),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if (candidate / "tools" / "inference" / "test_q1_student_video.py").exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve backend repo. Set --backend-repo or MOBILEOV_BACKEND_REPO "
        "to a valid Omni-Video-smolvlm2 checkout."
    )


def default_generation_ckpt(backend_repo: Path) -> Path:
    return (
        backend_repo
        / "omni_ckpts"
        / "hf_mobile_ov"
        / "stage1_joint_openvid_fullmobile_o_fulldit_diffonly_initlatest_bs64_v2_20260429_8gpu_60k.pt"
    )


def default_smolvlm2_ckpt(backend_repo: Path) -> Path:
    return backend_repo / "omni_ckpts" / "smolvlm2_500m" / "smolvlm2_500m.pt"


def resolve_path(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def build_backend_env(backend_repo: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(backend_repo) if not existing else f"{backend_repo}:{existing}"
    return env


def run_backend_python(
    backend_repo: Path,
    script_relpath: str,
    argv: Iterable[str],
) -> int:
    script_path = backend_repo / script_relpath
    if not script_path.exists():
        raise FileNotFoundError(f"Backend script not found: {script_path}")

    command: List[str] = [sys.executable, str(script_path), *list(argv)]
    print("Running backend command:")
    print(shlex.join(command))
    completed = subprocess.run(
        command,
        cwd=str(backend_repo),
        env=build_backend_env(backend_repo),
        check=False,
    )
    return int(completed.returncode)

