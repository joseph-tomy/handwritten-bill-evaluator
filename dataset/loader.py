"""Dataset loading helpers for handwritten bill evaluation.

This module keeps the dataset contract small and explicit:
- bill images live in `dataset/images`
- reference labels live in `dataset/ground_truth.json`
- each loaded sample is returned as a dictionary with `image_path` and `ground_truth`
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable


SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


class DatasetLoaderError(ValueError):
    """Raised when the dataset folder or ground-truth file is invalid."""


@dataclass(frozen=True)
class DatasetSample:
    """Internal representation of a loaded dataset sample."""

    image_path: Path
    ground_truth: dict[str, Any]


def load_dataset(dataset_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Load image samples and their matching ground truth records.

    Parameters
    ----------
    dataset_dir:
        Root dataset directory. Defaults to the `dataset/` folder next to this file.

    Returns
    -------
    list[dict[str, Any]]
        A list of records shaped like:
        [{"image_path": "...", "ground_truth": {...}}]
    """

    root_dir = _resolve_dataset_dir(dataset_dir)
    images_dir = root_dir / "images"
    ground_truth_path = root_dir / "ground_truth.json"

    _validate_dataset_root(root_dir, images_dir, ground_truth_path)

    ground_truth_index = _load_ground_truth_index(ground_truth_path)
    samples: list[DatasetSample] = []

    for image_path in _iter_image_files(images_dir):
        _validate_image_path(image_path)
        ground_truth = _resolve_ground_truth(image_path, ground_truth_index)
        samples.append(DatasetSample(image_path=image_path, ground_truth=ground_truth))

    return [
        {"image_path": str(sample.image_path), "ground_truth": sample.ground_truth}
        for sample in samples
    ]


def _resolve_dataset_dir(dataset_dir: Path | str | None) -> Path:
    """Resolve the dataset root to an absolute Path."""

    if dataset_dir is None:
        return Path(__file__).resolve().parent
    return Path(dataset_dir).expanduser().resolve()


def _validate_dataset_root(root_dir: Path, images_dir: Path, ground_truth_path: Path) -> None:
    """Validate the required dataset folders and files before loading."""

    if not root_dir.exists():
        raise DatasetLoaderError(f"Dataset directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise DatasetLoaderError(f"Dataset path is not a directory: {root_dir}")
    if not images_dir.exists():
        raise DatasetLoaderError(f"Images directory does not exist: {images_dir}")
    if not images_dir.is_dir():
        raise DatasetLoaderError(f"Images path is not a directory: {images_dir}")
    if not ground_truth_path.exists():
        raise DatasetLoaderError(f"Ground truth file does not exist: {ground_truth_path}")
    if not ground_truth_path.is_file():
        raise DatasetLoaderError(f"Ground truth path is not a file: {ground_truth_path}")


def _iter_image_files(images_dir: Path) -> Iterable[Path]:
    """Yield all supported image files in the dataset image folder.

    Unsupported file extensions are rejected immediately so the dataset does not
    silently ignore incorrect uploads.
    """

    for entry in sorted(images_dir.iterdir()):
        if entry.is_dir():
            # Nested directories are not part of the dataset contract for this loader.
            continue
        if entry.name.startswith("."):
            # Ignore repository placeholder files such as .gitkeep.
            continue
        if not entry.exists():
            raise DatasetLoaderError(f"Image file does not exist: {entry}")
        if entry.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise DatasetLoaderError(
                f"Unsupported image extension '{entry.suffix}' for file: {entry}. "
                f"Supported extensions are: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            )
        yield entry


def _validate_image_path(image_path: Path) -> None:
    """Validate a single image path before it is added to the result list."""

    if not image_path.exists():
        raise DatasetLoaderError(f"Image file does not exist: {image_path}")
    if not image_path.is_file():
        raise DatasetLoaderError(f"Image path is not a file: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise DatasetLoaderError(
            f"Unsupported image extension '{image_path.suffix}' for file: {image_path}. "
            f"Supported extensions are: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )


def _load_ground_truth_index(ground_truth_path: Path) -> dict[str, dict[str, Any]]:
    """Load ground-truth annotations and index them by image filename.

    Supported JSON shapes:
    - {"samples": [{"image_name": "bill1.jpg", "ground_truth": {...}}, ...]}
    - {"bill1.jpg": {...}, "bill2.png": {...}}
    """

    with ground_truth_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "samples" in payload:
        return _index_samples_list(payload["samples"], ground_truth_path)
    if isinstance(payload, dict):
        return {
            _normalize_image_key(str(image_name)): _ensure_ground_truth_object(
                ground_truth, ground_truth_path
            )
            for image_name, ground_truth in payload.items()
        }

    raise DatasetLoaderError(
        f"Ground truth file must contain a JSON object: {ground_truth_path}"
    )


def _index_samples_list(samples: Any, ground_truth_path: Path) -> dict[str, dict[str, Any]]:
    """Index a list-based ground truth payload by image filename."""

    if not isinstance(samples, list):
        raise DatasetLoaderError(
            f"The 'samples' field must be a list in {ground_truth_path}"
        )

    indexed_samples: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise DatasetLoaderError(
                f"Each sample must be a JSON object in {ground_truth_path}"
            )

        image_name = sample.get("image_name") or sample.get("image_path") or sample.get("image")
        if not image_name:
            raise DatasetLoaderError(
                f"Each sample must define 'image_name' in {ground_truth_path}"
            )

        truth = sample.get("ground_truth", sample.get("label", sample.get("annotations")))
        if truth is None:
            truth = {key: value for key, value in sample.items() if key not in {"image_name", "image_path", "image", "ground_truth", "label", "annotations"}}

        indexed_samples[_normalize_image_key(str(image_name))] = _ensure_ground_truth_object(
            truth, ground_truth_path
        )

    return indexed_samples


def _ensure_ground_truth_object(
    ground_truth: Any, ground_truth_path: Path
) -> dict[str, Any]:
    """Ensure each ground-truth record is a dictionary of annotations."""

    if not isinstance(ground_truth, dict):
        raise DatasetLoaderError(
            f"Ground truth entries must be JSON objects in {ground_truth_path}"
        )
    return ground_truth


def _normalize_image_key(image_name: str) -> str:
    """Normalize an image reference to a lowercase filename for matching."""

    return Path(image_name).name.lower()


def _resolve_ground_truth(image_path: Path, index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Match an image file to its ground-truth annotations."""

    image_key = _normalize_image_key(image_path.name)
    if image_key not in index:
        raise DatasetLoaderError(
            f"Missing ground truth for image: {image_path.name}. "
            f"Add a matching record to ground_truth.json."
        )
    return index[image_key]


def test_dataset_loader() -> None:
    """Small self-contained test for the dataset loader.

    This function creates a temporary dataset with one image and one matching
    ground-truth entry, then verifies that the loader returns one record.
    """

    with TemporaryDirectory() as temp_dir:
        root_dir = Path(temp_dir)
        images_dir = root_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        image_path = images_dir / "bill_001.jpg"
        image_path.write_bytes(b"fake-image-data")

        ground_truth_path = root_dir / "ground_truth.json"
        ground_truth_path.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "image_name": "bill_001.jpg",
                            "ground_truth": {
                                "vendor_name": "Demo Vendor",
                                "total_amount": 123.45,
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        records = load_dataset(root_dir)

        assert len(records) == 1, "Expected exactly one dataset record"
        assert records[0]["image_path"].endswith("bill_001.jpg"), "Image path mismatch"
        assert records[0]["ground_truth"]["vendor_name"] == "Demo Vendor", "Ground truth mismatch"
