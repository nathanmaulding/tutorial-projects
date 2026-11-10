"""Run model-written Python in a separate process.

What this protects against:
  * secrets: the child process gets a minimal environment with no API keys or tokens, so code the
    model writes cannot read or print your key (and so it can never end up in a saved trace)
  * hangs and crashes: a timeout kills runaway code; a crash only kills the child process
  * clutter: each run gets its own temporary working directory

What it does NOT do: block network access or the filesystem outside the temp dir. That is fine for
this tutorial (you are running a model you chose, on tasks you wrote), but for untrusted use run the
child inside a container or VM.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR")


def clean_env(workdir: str) -> dict:
    env = {k: os.environ[k] for k in SAFE_ENV_KEYS if k in os.environ}
    env.update({"HOME": workdir, "MPLBACKEND": "Agg", "PYTHONPATH": str(PROJECT_ROOT),
                "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"})
    return env


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


def run_python(code: str, files: dict[str, str | bytes] | None = None, timeout: float = 60.0,
               workdir: str | None = None) -> SandboxResult:
    """Run `code` in a fresh Python process. `files` maps file names to contents written first."""
    with tempfile.TemporaryDirectory(prefix="agentlab_") as tmp:
        wd = workdir or tmp
        for name, content in (files or {}).items():
            mode = "wb" if isinstance(content, bytes) else "w"
            with open(Path(wd) / name, mode) as f:
                f.write(content)
        script = Path(wd) / "_agent_code.py"
        script.write_text(code)
        try:
            proc = subprocess.run([sys.executable, str(script)], cwd=wd, env=clean_env(wd), capture_output=True, text=True,
                                  timeout=timeout)
            return SandboxResult(proc.stdout, proc.stderr, proc.returncode)
        except subprocess.TimeoutExpired as e:
            return SandboxResult(e.stdout or "", (e.stderr or "") + f"\nKilled after {timeout}s.", -9, True)
