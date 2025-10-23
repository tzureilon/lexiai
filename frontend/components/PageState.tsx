"use client";

import { ReactNode } from "react";

type PageStateProps = {
  title: string;
  description?: string;
  action?: ReactNode;
};

export function PageLoader({ message = "טוען את סביבת העבודה המאובטחת..." }: { message?: string }) {
  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-950/90 px-4 text-right text-slate-100">
      <div className="space-y-4 rounded-3xl border border-white/10 bg-slate-900/60 px-6 py-8 shadow-2xl">
        <div className="flex items-center justify-between gap-6">
          <span className="h-3 w-3 animate-ping rounded-full bg-blue-400" />
          <span className="text-sm font-semibold text-blue-100">LexiAI</span>
        </div>
        <p className="text-sm text-slate-200">{message}</p>
      </div>
    </div>
  );
}

export function PageEmptyState({ title, description, action }: PageStateProps) {
  return (
    <div className="rounded-3xl border border-dashed border-white/20 bg-white/5 px-6 py-10 text-right text-slate-200">
      <p className="text-lg font-semibold text-white">{title}</p>
      {description ? <p className="mt-3 max-w-2xl text-sm text-slate-300/90">{description}</p> : null}
      {action ? <div className="mt-6">{action}</div> : null}
    </div>
  );
}
