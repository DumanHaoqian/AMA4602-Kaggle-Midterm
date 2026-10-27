"""Explicit shared inventory for installation and release; never include exam data."""
from pathlib import Path

SKILL_NAME = "ama4602-kaggle-exam"
ROOT_FILES = (
    "SKILL.md", "README.md", "VERSION", "LICENSE", "THIRD_PARTY_NOTICES.md",
    "CHANGELOG.md", "CONTRIBUTING.md", ".gitignore", ".gitattributes",
    "requirements.txt", "requirements-reports.txt", "requirements-boosting.txt",
    "requirements-dev.txt",
)
TREES = {
    "agents": {".yaml"}, "scripts": {".py"}, "references": {".md"},
    "assets": {".md", ".json", ".sty", ".tex"},
    ".github/workflows": {".yml", ".yaml"},
}


def payload_files(root: Path) -> list[Path]:
    """Return relative paths from known source trees, refusing symbolic links."""
    selected = [root / name for name in ROOT_FILES]
    for name, suffixes in TREES.items():
        tree = root / name
        if tree.is_symlink():
            raise ValueError(f"Release source must not be a symlink: {tree}")
        if not tree.is_dir():
            raise ValueError(f"Missing release directory: {tree}")
        for path in tree.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Release source must not be a symlink: {path}")
            if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
                selected.append(path)
    for path in selected:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symbolic-link release file: {path}")
    return sorted(path.relative_to(root) for path in selected)
