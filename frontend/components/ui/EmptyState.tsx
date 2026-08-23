import type { ReactNode } from 'react';

import { Icon } from '@/components/ui/Icon';
import type { IconName } from '@/components/ui/Icon';

export interface EmptyStateProps {
  title: string;
  description?: string;
  icon?: IconName;
  /** Optional custom glyph overriding `icon`. */
  customIcon?: ReactNode;
}

/**
 * Placeholder content for reserved areas and empty collections.
 * Used to keep clean space for future modules without hard-coding them.
 */
export function EmptyState({ title, description, icon = 'chart', customIcon }: EmptyStateProps) {
  return (
    <div className="empty-state">
      <div className="empty-state__icon" aria-hidden="true">
        {customIcon ?? <Icon name={icon} size={26} />}
      </div>
      <p className="empty-state__title">{title}</p>
      {description === undefined ? null : (
        <p className="empty-state__description">{description}</p>
      )}
    </div>
  );
}
