import { StatusBadge } from '@/components/ui/StatusBadge';
import type { Availability } from '@/services/research/types';
import type { DatasetInfo, ModelInfo } from '@/services/research/types';

const AVAILABILITY_META: Record<Availability, { tone: 'success' | 'danger' | 'neutral'; label: string }> = {
  available: { tone: 'success', label: 'Available' },
  missing: { tone: 'danger', label: 'Not available' },
  unknown: { tone: 'neutral', label: 'Unknown' },
};

export function DatasetStatusCard({ dataset }: { dataset: DatasetInfo | undefined }) {
  if (dataset === undefined) {
    return null;
  }
  const meta = AVAILABILITY_META[dataset.status];
  return (
    <div className="dataset-card">
      <div className="dataset-card__head">
        <strong>{dataset.label}</strong>
        <StatusBadge tone={meta.tone}>{meta.label}</StatusBadge>
      </div>
      <dl className="kv-grid">
        <div className="kv-span">
          <dt>Root</dt>
          <dd className="mono muted small">{dataset.root}</dd>
        </div>
        <div>
          <dt>Subjects</dt>
          <dd className="mono">{dataset.numSubjects ?? '—'}</dd>
        </div>
        <div>
          <dt>Samples</dt>
          <dd className="mono">{dataset.numSamples?.toLocaleString() ?? '—'}</dd>
        </div>
        <div className="kv-span">
          <dt>Protocol</dt>
          <dd>{dataset.protocolNote}</dd>
        </div>
      </dl>
    </div>
  );
}

export function ModelStatusCard({ model }: { model: ModelInfo | undefined }) {
  if (model === undefined) {
    return null;
  }
  return (
    <div className="model-card">
      <div className="model-card__head">
        <strong>{model.label}</strong>
        <StatusBadge tone={model.implemented ? 'success' : 'info'}>
          {model.implemented ? 'Implemented' : 'Registered'}
        </StatusBadge>
      </div>
      <p className="muted small">{model.description}</p>
      <dl className="kv-grid">
        {Object.entries(model.inputs).map(([inputName, shape]) => (
          <div key={inputName}>
            <dt>{inputName}</dt>
            <dd className="mono">{shape.join('×')}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
