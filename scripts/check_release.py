#!/usr/bin/env python3
"""Validate standalone skill metadata, links, installer and release contents offline."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote
import zipfile

import yaml

from release_files import SKILL_NAME, payload_files


def run(root: Path, script: str, *args: str, success: bool = True) -> None:
    proc = subprocess.run([sys.executable, "-B", str(root / "scripts" / script), *args],
                          capture_output=True, text=True, timeout=60)
    if (proc.returncode == 0) != success:
        raise ValueError(f"Unexpected exit {proc.returncode} from {script}: {proc.stdout}\n{proc.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        files = payload_files(root)
        skill = (root / "SKILL.md").read_text(encoding="utf-8")
        match = re.match(r"\A---\n(.*?)\n---\n", skill, re.S)
        if not match:
            raise ValueError("SKILL.md needs YAML front matter.")
        meta = yaml.safe_load(match[1])
        if meta.get("name") != SKILL_NAME or not isinstance(meta.get("description"), str):
            raise ValueError("Skill name/description metadata is invalid.")
        if not 1 <= len(meta["description"]) <= 1024:
            raise ValueError("Skill description must contain 1–1024 characters.")
        interface = yaml.safe_load((root / "agents/openai.yaml").read_text(encoding="utf-8"))["interface"]
        if f"${SKILL_NAME}" not in interface.get("default_prompt", ""):
            raise ValueError("The default prompt must invoke the skill by name.")
        dataset_guides = sorted(path.relative_to(root) for path in (root / "datasets").glob("*/README.md"))
        for relative in files + dataset_guides:
            path = root / relative
            if path.suffix == ".py":
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative), feature_version=(3, 10))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import) and any(n.name == "hashlib" for n in node.names):
                        raise ValueError(f"Content digest calculation is not allowed: {relative}")
                    if isinstance(node, ast.ImportFrom) and node.module == "hashlib":
                        raise ValueError(f"Content digest calculation is not allowed: {relative}")
                    if isinstance(node, ast.Call):
                        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                        if name.lower() in ("sha256", "sha256_file", "sha1", "md5"):
                            raise ValueError(f"Content digest calculation is not allowed: {relative}")
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix in (".yml", ".yaml"):
                yaml.safe_load(path.read_text(encoding="utf-8"))
            elif path.suffix == ".md":
                text = path.read_text(encoding="utf-8")
                targets = re.findall(r"\]\(([^\s)]+)\)", text)
                targets += re.findall(r'''(?:href|src)=["']([^"']+)["']''', text, re.I)
                for target in targets:
                    target = unquote(target.split("#", 1)[0]).strip("<>")
                    if not target or "://" in target or target.startswith("mailto:"):
                        continue
                    resolved = (path.parent / target).resolve()
                    if root != resolved and root not in resolved.parents:
                        raise ValueError(f"Non-portable Markdown link in {relative}: {target}")
                    if not resolved.exists():
                        raise ValueError(f"Broken Markdown link in {relative}: {target}")
        print(f"PASS metadata, Python 3.10 syntax, JSON/YAML and local links ({len(files)} release files)")
        with tempfile.TemporaryDirectory(prefix="ama4602 release ") as temporary:
            base = Path(temporary)
            installed = base / "client skills" / SKILL_NAME
            run(root, "install_skill.py", "--dest", str(installed))
            if (installed / "SKILL.md").read_bytes() != (root / "SKILL.md").read_bytes():
                raise ValueError("Installed skill entrypoint differs from the source.")
            marker = installed / "keep-me.txt"
            marker.write_text("existing installation", encoding="utf-8")
            run(root, "install_skill.py", "--dest", str(installed), success=False)
            if marker.read_text(encoding="utf-8") != "existing installation":
                raise ValueError("Installer modified an existing installation.")
            output = base / "release archives"
            run(root, "build_release.py", "--output-dir", str(output))
            archives = list(output.glob("*.zip"))
            if len(archives) != 1:
                raise ValueError("Expected one release archive.")
            with zipfile.ZipFile(archives[0]) as bundle:
                expected = {f"{SKILL_NAME}/{p.as_posix()}" for p in files}
                if set(bundle.namelist()) != expected:
                    raise ValueError("ZIP inventory differs from the release inventory.")
                forbidden = {".git", ".venv", "data", "datasets", "dist", "__pycache__"}
                if any(forbidden.intersection(Path(name).parts) for name in bundle.namelist()):
                    raise ValueError("ZIP contains local data, an environment or build output.")
                if bundle.read(f"{SKILL_NAME}/SKILL.md") != (root / "SKILL.md").read_bytes():
                    raise ValueError("ZIP skill entrypoint differs from the source.")
            size = archives[0].stat().st_size
            run(root, "build_release.py", "--output-dir", str(output), success=False)
            if archives[0].stat().st_size != size:
                raise ValueError("Build overwrote an existing archive.")
        print("PASS installation with spaces, existing-install protection and clean release ZIP")
        return 0
    except (OSError, ValueError, SyntaxError, yaml.YAMLError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
