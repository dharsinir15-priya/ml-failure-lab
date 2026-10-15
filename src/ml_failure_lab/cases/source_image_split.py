"""Augmented images from one source must stay in the same split."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageSample:
    source_id: int
    image: np.ndarray
    label: int


def samples() -> list[ImageSample]:
    """Create asymmetric images and their flipped augmentations."""
    rows: list[ImageSample] = []

    for source_id in range(6):
        image = np.zeros((4, 4), dtype=np.uint8)
        image[0, 0] = source_id + 1
        image[1, 2] = source_id + 2
        image[3, 1] = source_id + 3

        rows.append(ImageSample(source_id, image, source_id % 2))
        rows.append(
            ImageSample(
                source_id,
                np.fliplr(image).copy(),
                source_id % 2,
            )
        )

    return rows


def random_row_split(
    rows: list[ImageSample],
) -> tuple[list[ImageSample], list[ImageSample]]:
    """Incorrectly split augmented rows independently."""
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(rows))
    midpoint = len(rows) // 2

    train = [rows[i] for i in indices[:midpoint]]
    test = [rows[i] for i in indices[midpoint:]]
    return train, test


def source_split(
    rows: list[ImageSample],
) -> tuple[list[ImageSample], list[ImageSample]]:
    """Keep all versions of each source in one split."""
    source_ids = sorted({row.source_id for row in rows})
    rng = np.random.default_rng(42)
    shuffled_ids = rng.permutation(source_ids)
    midpoint = len(shuffled_ids) // 2
    train_ids = set(shuffled_ids[:midpoint].tolist())

    train = [row for row in rows if row.source_id in train_ids]
    test = [row for row in rows if row.source_id not in train_ids]
    return train, test


def overlapping_sources(
    train: list[ImageSample],
    test: list[ImageSample],
) -> set[int]:
    """Return source IDs occurring in both train and test."""
    train_ids = {row.source_id for row in train}
    test_ids = {row.source_id for row in test}
    return train_ids & test_ids


def require_new_sources(
    train: list[ImageSample],
    test: list[ImageSample],
) -> None:
    """Require train and test source IDs to be disjoint."""
    overlap = overlapping_sources(train, test)

    if overlap:
        raise ValueError(
            "Evaluation of new source images requires disjoint "
            f"train and test source IDs; shared IDs: {sorted(overlap)}."
        )


def run() -> dict:
    """Compare a random row split with a source-based split."""
    measurements = []

    for approach, split in [
        ("bad", random_row_split),
        ("fixed", source_split),
    ]:
        train, test = split(samples())
        measurements.append(
            {
                "approach": approach,
                "train_images": len(train),
                "test_images": len(test),
                "overlapping_source_ids": len(overlapping_sources(train, test)),
            }
        )

    return {
        "case": "source-image-split",
        "question": (
            "Does this evaluation measure performance on previously unseen source images?"
        ),
        "measurements": measurements,
        "requirement": (
            "Train and test source IDs must be disjoint when evaluating new source images."
        ),
        "interpretation": (
            "Randomly splitting augmented images can place different "
            "versions of the same source image in both splits. "
            "Splitting by source ID keeps all versions of each source together."
        ),
        "limitation": (
            "This toy fixture demonstrates the split requirement only. "
            "A corrected split does not necessarily produce a higher evaluation score."
        ),
    }


def verify(approach: str) -> None:
    """Verify the selected split against the declared requirement."""
    split = random_row_split if approach == "bad" else source_split
    require_new_sources(*split(samples()))
