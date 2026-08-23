import { useState } from 'react';

import { StatusBadge } from '@/components/ui/StatusBadge';
import type {
  DatasetInfo,
  ExperimentSpec,
  ModelInfo,
} from '@/services/research/types';

export interface ExperimentConfigPanelProps {
  datasets: readonly DatasetInfo[];
  models: readonly ModelInfo[];
  simulated: boolean;
  busy: boolean;
  onSubmit: (spec: ExperimentSpec) => void;
  submitError?: string | null;
}

const FACE_SIZES = [224];
const EXPAND_RATIOS = [2.0, 1.6, 1.2];

function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9-_]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48);
}

/**
 * Experiment definition form. Pure presentation + local draft state;
 * submission goes through the ResearchService contract.
 */
export function ExperimentConfigPanel({
  datasets,
  models,
  simulated,
  busy,
  onSubmit,
  submitError,
}: ExperimentConfigPanelProps) {
  const [name, setName] = useState('resnet50-lopo-baseline');
  const [datasetId, setDatasetId] = useState(datasets[0]?.id ?? '');
  const [modelName, setModelName] = useState(models[0]?.name ?? '');
  const [faceSize, setFaceSize] = useState(224);
  const [expandRatio, setExpandRatio] = useState(2.0);
  const [imagenet, setImagenet] = useState(true);
  const [seed, setSeed] = useState(0);

  const nameValid = /^[a-z0-9][a-z0-9_-]{2,47}$/.test(name);
  const canSubmit =
    !busy && nameValid && datasetId !== '' && modelName !== '';

  function handleSubmit() {
    if (!canSubmit) {
      return;
    }
    const spec: ExperimentSpec = {
      name: slugify(name),
      datasetId,
      modelName,
      preprocessing: {
        recipe: 'gazehub',
        faceSize,
        expandRatio,
        imagenetNormalization: imagenet,
      },
      protocol: { type: 'lopo', seed },
      metric: 'mean_angular_error_deg',
    };
    onSubmit(spec);
  }

  return (
    <form
      className="experiment-form"
      onSubmit={(event) => {
        event.preventDefault();
        handleSubmit();
      }}
    >
      <div className="field">
        <label className="field__label" htmlFor="exp-name">
          Experiment name
        </label>
        <input
          id="exp-name"
          className={`input mono${nameValid ? '' : ' input--invalid'}`}
          value={name}
          onChange={(event) => setName(event.target.value)}
          aria-invalid={!nameValid}
          placeholder="lowercase-with-dashes"
        />
        {!nameValid ? <p className="hint">3–48 chars: a-z, 0-9, dashes.</p> : null}
      </div>

      <div className="form-row">
        <div className="field">
          <label className="field__label" htmlFor="exp-dataset">
            Dataset
          </label>
          <select
            id="exp-dataset"
            className="select"
            value={datasetId}
            onChange={(event) => setDatasetId(event.target.value)}
          >
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.label}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label className="field__label" htmlFor="exp-model">
            Model
          </label>
          <select
            id="exp-model"
            className="select"
            value={modelName}
            onChange={(event) => setModelName(event.target.value)}
          >
            {models.map((model) => (
              <option key={model.name} value={model.name}>
                {model.label}
                {model.implemented ? '' : ' (registered)'}
              </option>
            ))}
          </select>
        </div>
      </div>

      <fieldset className="fieldset">
        <legend className="field__label">Preprocessing · GazeHub</legend>
        <div className="form-row">
          <div className="field">
            <label className="field__label" htmlFor="exp-face-size">
              Face size
            </label>
            <select
              id="exp-face-size"
              className="select"
              value={faceSize}
              onChange={(event) => setFaceSize(Number(event.target.value))}
            >
              {FACE_SIZES.map((size) => (
                <option key={size} value={size}>
                  {size}×{size}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label className="field__label" htmlFor="exp-expand">
              Bbox expand ratio
            </label>
            <select
              id="exp-expand"
              className="select"
              value={expandRatio}
              onChange={(event) => setExpandRatio(Number(event.target.value))}
            >
              {EXPAND_RATIOS.map((ratio) => (
                <option key={ratio} value={ratio}>
                  ×{ratio.toFixed(1)}
                </option>
              ))}
            </select>
          </div>
          <label className="check-field">
            <input
              type="checkbox"
              checked={imagenet}
              onChange={(event) => setImagenet(event.target.checked)}
            />
            ImageNet normalization
          </label>
        </div>
      </fieldset>

      <fieldset className="fieldset">
        <legend className="field__label">Protocol</legend>
        <div className="form-row">
          <div className="field">
            <span className="field__label">Scheme</span>
            <p className="mono protocol-fixed">leave-one-person-out · 15 subjects</p>
          </div>
          <div className="field">
            <label className="field__label" htmlFor="exp-seed">
              Seed
            </label>
            <input
              id="exp-seed"
              type="number"
              min={0}
              max={4294967295}
              className="input mono"
              value={seed}
              onChange={(event) =>
                setSeed(Math.max(0, Math.floor(Number(event.target.value) || 0)))
              }
            />
          </div>
        </div>
      </fieldset>

      <div className="experiment-form__footer">
        <button type="submit" className="btn btn--primary btn--lg" disabled={!canSubmit}>
          {busy ? 'Submitting…' : 'Start Experiment'}
        </button>
        {simulated ? (
          <StatusBadge tone="info" pulse>
            Simulated run — training backend not connected
          </StatusBadge>
        ) : null}
        {submitError ? (
          <p className="form-error" role="alert">
            {submitError}
          </p>
        ) : null}
      </div>
    </form>
  );
}
