import type { ReactNode } from 'react';

export interface CardProps {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}

/** Bordered surface panel with optional header and action area. */
export function Card({ title, subtitle, actions, children, className }: CardProps) {
  const hasHeader = title !== undefined || actions !== undefined;
  return (
    <section className={className === undefined ? 'card' : `card ${className}`}>
      {hasHeader ? (
        <header className="card__header">
          <div>
            {title === undefined ? null : <h2 className="card__title">{title}</h2>}
            {subtitle === undefined ? null : <p className="card__subtitle">{subtitle}</p>}
          </div>
          {actions === undefined ? null : <div className="card__actions">{actions}</div>}
        </header>
      ) : null}
      <div className="card__body">{children}</div>
    </section>
  );
}
