"use client";

import { FormEvent, useMemo, useState } from "react";

import { AppShell } from "../components/AppShell";
import { PageEmptyState, PageLoader } from "../components/PageState";
import { apiFetch } from "../lib/api";
import { useRequireAuth } from "../lib/auth";

interface PredictionResponse {
  probability: number;
  rationale: string;
  recommended_actions: string[];
  created_at: string;
  signals: PredictionSignal[];
  quality_warnings: string[];
}

interface PredictionSignal {
  label: string;
  weight: number;
  direction: "positive" | "negative";
  evidence?: string | null;
}

export default function PredictPage() {
  const { session, loading } = useRequireAuth();
  const [details, setDetails] = useState("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!details.trim() || !session) return;
    setIsLoading(true);
    setError(null);
    try {
      const data = await apiFetch<PredictionResponse>("/predict", {
        method: "POST",
        body: JSON.stringify({ tenant_id: session.tenantId, user_id: session.userId, case_details: details }),
      });
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "שגיאה בחישוב החיזוי");
    } finally {
      setIsLoading(false);
    }
  };

  const confidenceLabel = useMemo(() => {
    if (!result) return "";
    if (result.probability >= 0.7) return "סיכוי גבוה";
    if (result.probability >= 0.4) return "סיכוי בינוני";
    return "סיכוי נמוך";
  }, [result]);

  if (loading || !session) {
    return <PageLoader message="טוען את מודל החיזוי המתקדם..." />;
  }

  const hero = (
    <div className="grid gap-4 md:grid-cols-3">
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">תוצאות אחרונות</p>
        <p className="mt-2 text-2xl font-bold text-white">{result ? `${Math.round(result.probability * 100)}%` : "—"}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">הערכת הצלחה מבוססת אותות ML.</p>
      </div>
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">בקרות איכות</p>
        <p className="mt-2 text-2xl font-bold text-white">{result?.quality_warnings.length ?? 0}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">התראות שיש לבדוק לפני הסתמכות מלאה.</p>
      </div>
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">אותות שנמצאו</p>
        <p className="mt-2 text-2xl font-bold text-white">{result?.signals.length ?? 0}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">ביטויים שחיזקו או החלישו את החיזוי.</p>
      </div>
    </div>
  );

  return (
    <AppShell
      title="חיזוי תיקים מוסבר"
      subtitle="מודל ML שנשען על קורפוסים משפטיים ומחזיר הסתברות, אותות מפורשים וצעדים אופרטיביים להמשך."
      hero={hero}
    >
      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <article className="glass-panel space-y-6">
          <header className="space-y-2 text-right">
            <p className="text-sm font-semibold text-white">הזינו את פרטי התיק</p>
            <p className="text-xs text-slate-200/80">
              ציינו עובדות, מועדים, גורמים מעורבים ויעדים משפטיים כדי לקבל הערכה מוסברת עם אותות מחזקים ומחלישים.
            </p>
          </header>
          <form className="space-y-4" onSubmit={handleSubmit}>
            <textarea
              id="case-details"
              className="h-48 w-full rounded-3xl border border-white/15 bg-slate-950/50 p-4 text-sm text-white placeholder:text-slate-400 focus:border-blue-300 focus:outline-none"
              value={details}
              onChange={(event) => setDetails(event.target.value)}
              placeholder="לדוגמה: תאר את השתלשלות האירועים, ההסכמות, המסמכים המרכזיים והמדדים המשפטיים להצלחה..."
            />
            <div className="flex flex-wrap justify-end gap-3 text-sm">
              <button
                type="button"
                onClick={() => {
                  setDetails("");
                  setResult(null);
                }}
                className="rounded-full border border-white/20 px-5 py-2 font-semibold text-slate-200 transition hover:border-white/40 hover:bg-white/10"
                disabled={!details && !result}
              >
                איפוס
              </button>
              <button
                type="submit"
                className="rounded-full bg-blue-500 px-6 py-2 font-semibold text-white shadow-[0_10px_30px_rgba(59,130,246,0.35)] transition hover:bg-blue-600"
                disabled={isLoading}
              >
                {isLoading ? "מחשב..." : "חשב חיזוי"}
              </button>
            </div>
          </form>
          {error ? <p className="text-sm text-red-300">{error}</p> : null}
          {!result ? (
            <PageEmptyState
              title="טרם חושב חיזוי"
              description="לאחר שתזינו פרטים מפורטים על התיק, LexiAI תחשב את סיכויי ההצלחה ותציג רציונל ואותות מרכזיים."
            />
          ) : null}
        </article>

        <aside className="space-y-6">
          {result ? (
            <div className="glass-panel space-y-6">
              <div className="flex flex-col gap-3 text-right">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-white">תוצאה משוערת</h2>
                  <span className="rounded-full border border-white/20 px-3 py-1 text-xs font-semibold text-white/80">
                    {confidenceLabel}
                  </span>
                </div>
                <ConfidenceIndicator probability={result.probability} />
                <p className="text-sm leading-relaxed text-slate-100/90 whitespace-pre-line">{result.rationale}</p>
                <time className="text-[0.7rem] text-slate-300/70">עודכן: {new Date(result.created_at).toLocaleString()}</time>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <SignalList
                  title="מחזקים"
                  tone="positive"
                  signals={result.signals.filter((signal) => signal.direction === "positive")}
                />
                <SignalList
                  title="מחלישים"
                  tone="negative"
                  signals={result.signals.filter((signal) => signal.direction === "negative")}
                />
              </div>

              <div className="rounded-3xl border border-white/10 bg-slate-950/40 p-4">
                <h3 className="text-sm font-semibold text-white">צעדים מומלצים</h3>
                <ul className="mt-3 space-y-2 text-sm text-slate-100/90">
                  {result.recommended_actions.map((action, index) => (
                    <li key={`${action}-${index}`} className="flex items-start gap-2 rounded-2xl border border-white/10 bg-white/5 p-3">
                      <span className="mt-1 inline-block h-2 w-2 rounded-full bg-blue-400" aria-hidden />
                      <span className="leading-relaxed">{action}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {result.quality_warnings.length > 0 ? (
                <div className="rounded-3xl border border-amber-200/60 bg-amber-500/10 p-4 text-xs text-amber-100">
                  <h3 className="text-sm font-semibold text-amber-200">בקרות איכות</h3>
                  <ul className="mt-2 space-y-1">
                    {result.quality_warnings.map((warning, index) => (
                      <li key={`${warning}-${index}`}>{warning}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          ) : (
            <PageEmptyState
              title="המתן לתוצאה"
              description="לאחר חישוב תקבל כאן את ההסתברות, הרציונל, רשימת האותות והצעדים האופרטיביים להמשך."
            />
          )}
        </aside>
      </section>
    </AppShell>
  );
}

function SignalList({ title, signals, tone }: { title: string; signals: PredictionSignal[]; tone: "positive" | "negative" }) {
  const toneClasses =
    tone === "positive"
      ? "border-emerald-300/40 bg-emerald-500/10 text-emerald-100"
      : "border-rose-300/40 bg-rose-500/10 text-rose-100";

  if (signals.length === 0) {
    return (
      <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-xs text-slate-300">
        <h4 className="text-sm font-semibold text-white">{title}</h4>
        <p className="mt-2">לא זוהו אותות.</p>
      </div>
    );
  }

  return (
    <div className={`rounded-2xl border p-4 ${toneClasses}`}>
      <h4 className="text-sm font-semibold text-white">{title}</h4>
      <ul className="mt-3 space-y-2 text-xs">
        {signals.map((signal, index) => (
          <li key={`${signal.label}-${index}`} className="flex items-center justify-between gap-3">
            <span className="font-semibold">{signal.label}</span>
            <span className="rounded-full bg-white/20 px-2 py-1 text-[0.65rem] font-semibold text-white">
              {signal.weight.toFixed(2)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ConfidenceIndicator({ probability }: { probability: number }) {
  const percentage = Math.round(probability * 100);
  return (
    <div className="space-y-2 text-right">
      <div className="flex items-center justify-between text-xs text-slate-200/80">
        <span>סיכויי הצלחה</span>
        <span>{percentage}%</span>
      </div>
      <div className="h-3 w-full rounded-full bg-white/10">
        <div className="h-full rounded-full bg-gradient-to-l from-blue-500 to-blue-700" style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}
