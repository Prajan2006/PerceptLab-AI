"""Unit tests for the RAW MPIIFaceGaze layout adapter (synthetic fixture)."""

import pytest

from data.mpiifacegaze.raw_adapter import (
    AnnotationParseError,
    discover_raw_subjects,
    parse_annotation_line,
    read_subject_annotations,
    validate_raw_dataset,
)
from data.mpiifacegaze.raw_synthetic import (
    build_synthetic_raw_dataset,
    build_synthetic_raw_subject,
    format_annotation_row,
)
from data.splits import lopo_folds


@pytest.fixture()
def raw_root(tmp_path):
    return build_synthetic_raw_dataset(tmp_path / "Data", subjects=("p00", "p01"))


class TestSubjectDiscovery:
    def test_discovers_sorted_subjects_ignoring_artifacts(self, raw_root):
        assert discover_raw_subjects(raw_root) == ["p00", "p01"]

    def test_artifact_directories_never_match(self, tmp_path):
        root = tmp_path / "D"
        (root / "__MACOSX" / "p42").mkdir(parents=True)
        (root / "notes").mkdir()
        (root / "p05").mkdir()
        assert discover_raw_subjects(root) == ["p05"]

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            discover_raw_subjects(tmp_path / "nope")


class TestAnnotationParsing:
    LINE = format_annotation_row("day01", "0005.jpg", index=3)

    def test_row_has_twenty_eight_tokens(self):
        assert len(self.LINE.split()) == 28

    def test_parse_preserves_field_groups(self, tmp_path):
        annotation = parse_annotation_line(self.LINE, tmp_path / "p07", 1)

        assert annotation.subject_id == "p07"
        assert annotation.session_id == "day01"
        assert annotation.relative_path == "day01/0005.jpg"
        assert str(annotation.image_path).endswith("day01\\0005.jpg") or str(
            annotation.image_path
        ).endswith("day01/0005.jpg")
        # fields 2-3: gaze location on screen (pixels)
        assert annotation.gaze_screen_location == (503.0, 303.0)
        # fields 4-15: six facial landmarks, dataset order
        assert len(annotation.landmarks) == 6
        assert annotation.landmarks[0] == (100.0, 200.0)
        assert annotation.landmarks[5] == (150.0, 225.0)
        # fields 16-21: head pose = rotation + translation
        assert annotation.head_rotation[0] == pytest.approx(-0.232179 + 3 * 0.001)
        assert annotation.head_rotation[1:] == (pytest.approx(0.055685), pytest.approx(0.018205))
        assert annotation.head_translation == (
            pytest.approx(28.351504),
            pytest.approx(1.174807),
            pytest.approx(529.783734),
        )
        # fields 22-24: face centre (fc) — raw camera coordinates preserved
        assert annotation.face_center == (
            pytest.approx(27.792112),
            pytest.approx(23.422692),
            pytest.approx(524.537075),
        )
        # fields 25-27: gaze target (gt) — raw values preserved verbatim
        assert annotation.gaze_target == (
            pytest.approx(11.040978),
            pytest.approx(166.869249),
            pytest.approx(-27.728178),
        )
        # field 28
        assert annotation.eye_side == "left"
        assert annotation.source_line == 1

    def test_parser_does_not_derive_or_normalize(self, tmp_path):
        """Parsing stays normalization-free; direction derivation is preprocessing."""
        import dataclasses

        import data.mpiifacegaze.raw_adapter as raw_module

        field_names = {f.name for f in dataclasses.fields(raw_module.RawAnnotation)}
        assert field_names == {
            "subject_id",
            "session_id",
            "relative_path",
            "image_path",
            "gaze_screen_location",
            "landmarks",
            "head_rotation",
            "head_translation",
            "face_center",
            "gaze_target",
            "eye_side",
            "source_line",
        }
        assert not hasattr(raw_module, "derive_gaze_direction")
        assert not hasattr(raw_module, "normalize")

    def test_official_layout_documented_in_module(self, tmp_path):
        import data.mpiifacegaze.raw_adapter as raw_module

        doc = raw_module.__doc__ or ""
        assert "gaze direction = gt - fc" in doc
        assert "rotation" in doc and "translation" in doc

    def test_wrong_token_count_rejected(self, tmp_path):
        with pytest.raises(AnnotationParseError) as excinfo:
            parse_annotation_line("day01/x.jpg 1 2 3", tmp_path, 4)
        assert "28" in str(excinfo.value)

    def test_invalid_eye_side_rejected(self, tmp_path):
        line = format_annotation_row("day01", "0005.jpg", eye_side="both")
        with pytest.raises(AnnotationParseError):
            parse_annotation_line(line, tmp_path, 1)

    def test_non_numeric_field_rejected(self, tmp_path):
        tokens = self.LINE.split()
        tokens[15] = "NaN-not"
        with pytest.raises(AnnotationParseError):
            parse_annotation_line(" ".join(tokens), tmp_path, 1)

    def test_bad_path_shape_rejected(self, tmp_path):
        tokens = self.LINE.split()
        tokens[0] = "day01/sub/0005.jpg"
        with pytest.raises(AnnotationParseError):
            parse_annotation_line(" ".join(tokens), tmp_path, 1)


class TestSubjectAnnotationReading:
    def test_valid_and_malformed_rows_reported(self, tmp_path):
        subject_dir = build_synthetic_raw_subject(tmp_path, "p00", include_malformed_row=True)
        annotations, malformed = read_subject_annotations(subject_dir)
        assert len(annotations) > 0
        assert len(malformed) == 1
        assert malformed[0].source_line > annotations[-1].source_line

    def test_missing_annotation_file_reported(self, tmp_path):
        empty = tmp_path / "p09"
        empty.mkdir()
        annotations, malformed = read_subject_annotations(empty)
        assert annotations == []
        assert "missing annotation file" in malformed[0].reason


class TestDatasetValidation:
    def test_full_fixture_counts(self, raw_root):
        report = validate_raw_dataset(raw_root)

        # p00: 1 session x 3 frames, first frame intentionally missing;
        # p01: 2 sessions x 3 frames + one malformed row (excluded from records).
        assert report.subjects == ["p00", "p01"]
        assert report.annotation_records == 9
        assert report.images_on_disk == 8
        assert report.valid_matches == report.images_on_disk
        assert report.missing_images_total == 1
        assert len(report.missing_images_sample) == 1
        assert len(report.malformed_annotations) == 1
        assert "expected 28" in report.malformed_annotations[0]
        assert report.ignored_artifacts >= 3  # __MACOSX subtree + .DS_Store files
        assert "p00/day01" in report.sessions
        assert "p01/day02" in report.sessions

    def test_every_annotation_image_exists_when_no_gaps(self, tmp_path):
        subject = build_synthetic_raw_subject(tmp_path, "p03", include_missing_image=False)
        report = validate_raw_dataset(subject.parent)
        assert report.annotation_records == 6
        assert report.missing_images_total == 0
        assert report.valid_matches == report.annotation_records


class TestLopoOnRawDiscovery:
    def test_fifteen_deterministic_folds(self, tmp_path):
        subjects = tuple(f"p{index:02d}" for index in range(15))
        for subject in subjects:
            build_synthetic_raw_subject(tmp_path, subject, sessions=("day01",), frames_per_session=1)

        discovered = discover_raw_subjects(tmp_path)
        folds = lopo_folds(discovered)

        assert len(folds) == 15
        held_out = [fold.test_subjects[0] for fold in folds]
        assert held_out == list(subjects)  # deterministic sorted order p00…p14
        for fold in folds:
            assert set(fold.test_subjects).isdisjoint(fold.train_subjects)
