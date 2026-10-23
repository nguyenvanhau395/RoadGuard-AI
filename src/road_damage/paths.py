"""Resolve the project independently of the notebook working directory."""

from pathlib import Path


DATASET_DIRECTORIES = {"crack": "crack-4", "pothole": "pothole-v2"}


def project_root(root: str | Path | None = None) -> Path:
    """Return the explicit project root, or the root containing this package."""
    return Path(root).expanduser().resolve() if root is not None else Path(__file__).resolve().parents[2]


def dataset_root(dataset: str, root: str | Path | None = None) -> Path:
    if dataset not in DATASET_DIRECTORIES:
        raise ValueError(f"Unknown dataset {dataset!r}; choose {tuple(DATASET_DIRECTORIES)}")
    return project_root(root) / "data" / "raw" / DATASET_DIRECTORIES[dataset]
