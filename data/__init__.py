"""Dataset layer adapters and utilities."""

from .config import (
    DATASET_REGISTRY_PATH,
    DatasetLocation,
    DatasetNotAvailableError,
    load_dataset_registry,
    resolve_dataset_root,
)
from .gaze3d import angular_error_deg, normalize_vector, yaw_pitch_to_gaze_deg
from .mpiifacegaze import (
    GazeBatch,
    GazeSample,
    MPIIFaceGazeDataset,
    RawMPIIFaceGazeDataset,
    build_index,
    build_lopo_fold_datasets,
    collate_gaze_samples,
    load_index,
    make_gaze_dataloader,
)
from .preprocessing import GazeHubPreprocessor, PreprocessedSample, PreprocessingProtocol, PreprocessConfig
from .splits import LOPOFold, load_lopo_folds, lopo_folds, save_lopo_folds

__all__ = [
    "DATASET_REGISTRY_PATH",
    "DatasetLocation",
    "DatasetNotAvailableError",
    "GazeBatch",
    "GazeHubPreprocessor",
    "GazeSample",
    "LOPOFold",
    "MPIIFaceGazeDataset",
    "PreprocessedSample",
    "PreprocessConfig",
    "PreprocessingProtocol",
    "RawMPIIFaceGazeDataset",
    "angular_error_deg",
    "build_index",
    "build_lopo_fold_datasets",
    "collate_gaze_samples",
    "load_dataset_registry",
    "load_index",
    "load_lopo_folds",
    "lopo_folds",
    "make_gaze_dataloader",
    "normalize_vector",
    "resolve_dataset_root",
    "save_lopo_folds",
    "yaw_pitch_to_gaze_deg",
]
