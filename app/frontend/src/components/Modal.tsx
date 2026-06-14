import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import type { ReactNode } from 'react';

export function Modal({ open, onClose, children, wide = false }: {
  open: boolean; onClose: () => void; children: ReactNode; wide?: boolean;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => { window.removeEventListener('keydown', onKey); document.body.style.overflow = ''; };
  }, [open, onClose]);
  if (!open) return null;
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" role="dialog" aria-modal="true">
      <div className="absolute inset-0 bg-black/70" onClick={onClose} />
      <div className={`relative panel p-6 max-h-[90vh] overflow-y-auto w-full ${wide ? 'max-w-4xl' : 'max-w-xl'}`}>
        <button onClick={onClose} aria-label="Close" className="absolute top-3 right-3 text-steel hover:text-fog"><X size={18} /></button>
        {children}
      </div>
    </div>,
    document.body,
  );
}
