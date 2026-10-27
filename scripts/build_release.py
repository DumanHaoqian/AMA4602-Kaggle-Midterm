#!/usr/bin/env python3
"""Build a versioned skill ZIP from the explicit release inventory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import zipfile

from release_files import SKILL_NAME, payload_files


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "dist")
    args = parser.parse_args()
    archive = None
    created = False
    try:
        version = (root / "VERSION").read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[a-z0-9.]+)?", version):
            raise ValueError("VERSION must contain a semantic version, e.g. 0.1.0.")
        files = payload_files(root)
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        archive = output / f"{SKILL_NAME}-v{version}.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
            created = True
            for relative in files:
                bundle.write(root / relative, f"{SKILL_NAME}/{relative.as_posix()}")
        print(json.dumps({"version": version, "archive": str(archive),
                          "files": len(files), "size_bytes": archive.stat().st_size}, indent=2))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if created and archive is not None:
            archive.unlink(missing_ok=True)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
