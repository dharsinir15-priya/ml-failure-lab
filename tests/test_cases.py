import pytest

from ml_failure_lab.cases import (
    feature_selection,
    group_leakage,
    minority_recall,
    source_image_split,
)
from ml_failure_lab.registry import CASES


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_bad_approach_violates_declared_requirement(case):
    with pytest.raises(ValueError):
        case.verify("bad")


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_fixed_approach_meets_declared_requirement(case):
    case.verify("fixed")


def test_group_fixture_exposes_recognition_instead_of_new_entity_prediction():
    result = group_leakage.run()["measurements"]
    assert result[0]["overlapping_entities"] == 40
    assert result[0]["accuracy"] == 1.0
    assert result[1]["overlapping_entities"] == 0
    assert result[1]["accuracy"] == 0.5


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_selector_audit_tracks_actual_test_rows_in_fit_input(seed):
    bad, test_ids, bad_score = feature_selection.experiment("bad", seed)
    fixed, fixed_test_ids, fixed_score = feature_selection.experiment("fixed", seed)

    assert len(bad.fitted_rows & set(test_ids)) == 50
    assert not fixed.fitted_rows & set(fixed_test_ids)
    assert len(fixed.fitted_rows) == 150
    assert 0 <= bad_score <= 1
    assert 0 <= fixed_score <= 1
    # A score ordering is deliberately not part of the evaluation requirement.


def test_minority_report_exposes_zero_detected_examples_without_changing_accuracy():
    bad = minority_recall.report("bad")
    fixed = minority_recall.report("fixed")

    assert bad["accuracy"] == fixed["accuracy"] == 0.99
    assert fixed["minority_recall"] == 0.0
    assert fixed["balanced_accuracy"] == 0.5
    assert fixed["actual_minority_records"] == 5
    assert fixed["detected_minority_records"] == 0


def test_source_image_fixture_rejects_random_split_and_accepts_source_split():
    rows = source_image_split.samples()

    bad_train, bad_test = source_image_split.random_row_split(rows)
    fixed_train, fixed_test = source_image_split.source_split(rows)

    # The incorrect random split must expose source-image leakage.
    assert source_image_split.overlapping_sources(bad_train, bad_test)

    # The corrected split must keep source IDs disjoint.
    assert source_image_split.overlapping_sources(fixed_train, fixed_test) == set()

    # Reject the incorrect approach and accept the corrected one.
    with pytest.raises(ValueError):
        source_image_split.verify("bad")

    source_image_split.verify("fixed")


def test_random_row_split_is_reproducible():
    rows = source_image_split.samples()

    train1, test1 = source_image_split.random_row_split(rows)
    train2, test2 = source_image_split.random_row_split(rows)

    assert [row.source_id for row in train1] == [row.source_id for row in train2]
    assert [row.source_id for row in test1] == [row.source_id for row in test2]


def test_flipping_asymmetric_images_changes_them():
    rows = source_image_split.samples()

    for i in range(0, len(rows), 2):
        original = rows[i].image
        flipped = rows[i + 1].image

        assert not (original == flipped).all()
