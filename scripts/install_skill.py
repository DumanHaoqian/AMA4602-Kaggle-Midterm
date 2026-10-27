#!/usr/bin/env python3
"""Copy this skill into a client skill directory without overwriting an install."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

from release_files import SKILL_NAME, payload_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", choices=("codex", "codex-legacy", "claude"), default="codex")
    parser.add_argument("--dest", type=Path, help="Exact destination skill folder; overrides --client.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bases = {
        "codex": Path.home() / ".agents" / "skills",
        "codex-legacy": Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "skills",
        "claude": Path.home() / ".claude" / "skills",
    }
    requested = args.dest or bases[args.client] / SKILL_NAME
    destination = requested.expanduser().absolute()
    created = False
    try:
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Destination already exists; keep or move it before installing: {destination}")
        resolved = destination.resolve()
        if resolved == root or root in resolved.parents or resolved in root.parents:
            raise ValueError("Destination must be outside the source skill tree and its parents.")
        files = payload_files(root)
        destination.mkdir(parents=True, exist_ok=False)
        created = True
        for relative in files:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, target)
        print(f"Installed {SKILL_NAME} ({len(files)} files) into {destination}")
        print("Open a new client session if the skill is not detected.")
        return 0
    except (OSError, ValueError) as exc:
        if created:
            shutil.rmtree(destination)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
