"""MPIIFaceGaze dataset layer (adapter, index, synthetic fixtures).

Two layouts are supported without coupling:

- ``adapter``      — the MATLAB evaluation subset (``pYY.mat``).
- ``raw_adapter``  — the raw verified layout (``pYY/dayXX/*.jpg`` +
  ``pYY/pYY.txt`` annotation rows) used by the downloaded dataset.
"""

from .adapter import GazeSample, MPIIFaceGazeDataset
from .metadata import build_index, load_index
from .raw_adapter import (
    AnnotationParseError,
    DatasetValidationReport,
    MalformedAnnotation,
    RawAnnotation,
    count_ignored_artifacts,
    discover_raw_subjects,
    parse_annotation_line,
    read_subject_annotations,
    validate_raw_dataset,
)
from .raw_dataset import (
    GazeBatch,
    RawMPIIFaceGazeDataset,
    assert_fold_disjointness,
    build_lopo_fold_datasets,
    collate_gaze_samples,
    make_gaze_dataloader,
)
from .synthetic import build_synthetic_dataset, build_synthetic_subject_file
from .raw_synthetic import (
    build_synthetic_raw_dataset,
    build_synthetic_raw_subject,
    format_annotation_row,
)

__all__ = [
    "AnnotationParseError",
    "DatasetValidationReport",
    "GazeBatch",
    "GazeSample",
    "MPIIFaceGazeDataset",
    "MalformedAnnotation",
    "RawAnnotation",
    "RawMPIIFaceGazeDataset",
    "assert_fold_disjointness",
    "build_index",
    "build_lopo_fold_datasets",
    "build_synthetic_dataset",
    "build_synthetic_raw_dataset",
    "build_synthetic_raw_subject",
    "build_synthetic_subject_file",
    "collate_gaze_samples",
    "count_ignored_artifacts",
    "discover_raw_subjects",
    "format_annotation_row",
    "load_index",
    "make_gaze_dataloader",
    "parse_annotation_line",
    "read_subject_annotations",
    "validate_raw_dataset",
]
