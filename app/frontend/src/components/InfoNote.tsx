import { useState, type ReactNode } from 'react';

/**
 * Collapsible "how this works" explainer shown under a section header.
 * Presentational only — pass a title and rich children (paragraphs / lists).
 */
export function InfoNote({
  title = 'How this works',
  children,
  defaultOpen = false,
}: {
  title?: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="panel border-l-2 border-l-accent/60 p-4">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 text-left"
        aria-expanded={open}
      >
        <span className="text-accent text-xs font-medium uppercase tracking-widest">
          ⓘ {title}
        </span>
        <span className="ml-auto text-xs text-steel">{open ? 'hide' : 'show'}</span>
      </button>
      {open && (
        <div className="mt-3 space-y-2 text-sm leading-relaxed text-steel">{children}</div>
      )}
    </div>
  );
}
