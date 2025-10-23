"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "../components/AppShell";
import { PageEmptyState, PageLoader } from "../components/PageState";
import { apiFetch } from "../lib/api";
import { useRequireAuth } from "../lib/auth";

interface ContextualReference {
  document_id: number;
  filename: string;
  snippet: string;
  score: number;
  explanation?: string;
}

interface WitnessQuestionSet {
  stage: string;
  questions: string[];
}

interface WitnessPlan {
  id: number;
  user_id: string;
  witness_name: string;
  witness_role: string;
  case_summary: string;
  strategy: string;
  focus_areas: string[];
  question_sets: WitnessQuestionSet[];
  risk_controls: string[];
  contextual_references: ContextualReference[];
  created_at: string;
  quality_notes: string[];
}

export default function WitnessPage() {
  const { session, loading } = useRequireAuth();
  const [witnessName, setWitnessName] = useState("");
  const [witnessRole, setWitnessRole] = useState("");
  const [caseSummary, setCaseSummary] = useState("");
  const [objectives, setObjectives] = useState("");
  const [plan, setPlan] = useState<WitnessPlan | null>(null);
  const [history, setHistory] = useState<WitnessPlan[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const objectiveList = useMemo(
    () =>
      objectives
        .split(/[\n,]/)
        .map((item) => item.trim())
        .filter(Boolean),
    [objectives],
  );

  const loadHistory = useCallback(async () => {
    if (!session) return;
    try {
      const data = await apiFetch<WitnessPlan[]>(`/witness/${session.userId}?tenant_id=${session.tenantId}`);
      setHistory(data);
      setPlan((prev) => prev ?? (data.length > 0 ? data[0] : null));
    } catch (err) {
      setError(err instanceof Error ? err.message : "שגיאה בטעינת תוכניות העדים");
    }
  }, [session]);

  useEffect(() => {
    if (!loading && session) {
      void loadHistory();
    }
  }, [loadHistory, loading, session]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!witnessName || !witnessRole || !caseSummary || !session) {
      setError("נא למלא את כל שדות החובה");
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const payload = await apiFetch<WitnessPlan>("/witness", {
        method: "POST",
        body: JSON.stringify({
          tenant_id: session.tenantId,
          user_id: session.userId,
          witness_name: witnessName,
          witness_role: witnessRole,
          case_summary: caseSummary,
          objectives: objectiveList,
        }),
      });
      setPlan(payload);
      await loadHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : "שגיאה ביצירת תוכנית החקירה");
    } finally {
      setIsLoading(false);
    }
  };

  if (loading || !session) {
    return <PageLoader message="טוען את סביבת חקירות העדים..." />;
  }

  const hero = (
    <div className="grid gap-4 md:grid-cols-3">
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">תוכניות שמורות</p>
        <p className="mt-2 text-2xl font-bold text-white">{history.length}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">נגישות מידית לכל צוות העד.</p>
      </div>
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">הקשרים ממסמכים</p>
        <p className="mt-2 text-2xl font-bold text-white">{plan?.contextual_references.length ?? 0}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">קטעי מסמכים המשולבים בתכנית.</p>
      </div>
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">מוקדי חקירה</p>
        <p className="mt-2 text-2xl font-bold text-white">{plan?.focus_areas.length ?? 0}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">נושאים לתעדוף במהלך החקירה.</p>
      </div>
    </div>
  );

  return (
    <AppShell
      title="חדר המלחמה לחקירות עדים"
      subtitle="צרו אסטרטגיה דינמית לעד, עם סטי שאלות, מוקדי חקירה, בקרות איכות והצלבה למסמכים שהעליתם."
      hero={hero}
    >
      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1.8fr)]">
        <div className="space-y-6">
          <form onSubmit={handleSubmit} className="glass-panel space-y-5">
            <header className="space-y-2 text-right">
              <p className="text-sm font-semibold text-white">הגדרת עד חדש</p>
              <p className="text-xs text-slate-200/80">מלאו את פרטי העד, תפקידו ויעדי החקירה כדי ליצור תכנית חכמה.</p>
            </header>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="witness-name">
                  שם העד
                </label>
                <input
                  id="witness-name"
                  value={witnessName}
                  onChange={(event) => setWitnessName(event.target.value)}
                  className="mt-2 w-full rounded-3xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white placeholder:text-slate-400 focus:border-blue-300 focus:outline-none"
                  placeholder="לדוגמה: רות כהן"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="witness-role">
                  תפקיד העד
                </label>
                <input
                  id="witness-role"
                  value={witnessRole}
                  onChange={(event) => setWitnessRole(event.target.value)}
                  className="mt-2 w-full rounded-3xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white placeholder:text-slate-400 focus:border-blue-300 focus:outline-none"
                  placeholder="עד מומחה, נציג חברה, לקוח"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="case-summary">
                  תקציר המקרה
                </label>
                <textarea
                  id="case-summary"
                  value={caseSummary}
                  onChange={(event) => setCaseSummary(event.target.value)}
                  className="mt-2 h-28 w-full rounded-3xl border border-white/20 bg-slate-950/50 p-3 text-sm text-white placeholder:text-slate-400 focus:border-blue-300 focus:outline-none"
                  placeholder="תאר את העובדות המרכזיות, הדגשים והנושאים שבמחלוקת..."
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="objectives">
                  יעדי החקירה
                </label>
                <textarea
                  id="objectives"
                  value={objectives}
                  onChange={(event) => setObjectives(event.target.value)}
                  className="mt-2 h-24 w-full rounded-3xl border border-white/20 bg-slate-950/50 p-3 text-sm text-white placeholder:text-slate-400 focus:border-blue-300 focus:outline-none"
                  placeholder="הפרידו בפסיקים או שורות: להעמיק בנוהלי החברה, להראות פערים במסמכים, להדגיש היסוס העד..."
                />
              </div>
            </div>
            <div className="flex flex-wrap justify-end gap-3 text-sm">
              <button
                type="button"
                onClick={() => {
                  setWitnessName("");
                  setWitnessRole("");
                  setCaseSummary("");
                  setObjectives("");
                }}
                className="rounded-full border border-white/20 px-5 py-2 font-semibold text-slate-200 transition hover:border-white/40 hover:bg-white/10"
              >
                איפוס
              </button>
              <button
                type="submit"
                className="rounded-full bg-blue-500 px-6 py-2 font-semibold text-white shadow-[0_10px_30px_rgba(59,130,246,0.35)] transition hover:bg-blue-600"
                disabled={isLoading}
              >
                {isLoading ? "מנתח..." : "צור תוכנית"}
              </button>
            </div>
            {error ? <p className="text-sm text-red-300">{error}</p> : null}
          </form>

          <div className="glass-panel">
            <h2 className="text-sm font-semibold text-white">תוכניות קודמות</h2>
            {history.length === 0 ? (
              <p className="mt-3 text-xs text-slate-200/70">לא נמצאו תוכניות שמורות למשתמש הנוכחי.</p>
            ) : (
              <ul className="mt-4 space-y-2 text-sm">
                {history.map((item) => (
                  <li key={item.id}>
                    <button
                      onClick={() => setPlan(item)}
                      className={`w-full rounded-3xl border px-4 py-3 text-right transition ${
                        plan?.id === item.id
                          ? "border-blue-300/60 bg-blue-500/20 text-white"
                          : "border-white/10 bg-white/5 text-slate-200 hover:border-white/25 hover:bg-white/10"
                      }`}
                    >
                      <div className="flex items-center justify-between text-[0.7rem] text-slate-200/80">
                        <span className="font-semibold text-white">{item.witness_name}</span>
                        <time>{new Date(item.created_at).toLocaleString()}</time>
                      </div>
                      <p className="mt-1 text-xs text-slate-200/70">{item.witness_role}</p>
                      <p className="mt-1 max-h-12 overflow-hidden text-xs text-slate-200/60">{item.case_summary}</p>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        <div className="glass-panel">
          {!plan ? (
            <PageEmptyState
              title="הזן פרטי עד כדי להתחיל"
              description="לאחר יצירת התכנית יופיעו כאן אסטרטגיית החקירה, סטי השאלות, בקרות הסיכונים והקשרים למסמכי הלקוח."
            />
          ) : (
            <article className="space-y-6 text-sm text-slate-100/90">
              <header className="space-y-2 border-b border-white/10 pb-4">
                <h2 className="text-2xl font-semibold text-white">תכנית חקירה עבור {plan.witness_name}</h2>
                <p className="text-xs text-slate-200/80">
                  תפקיד: {plan.witness_role} · נוצר: {new Date(plan.created_at).toLocaleString()}
                </p>
                <p className="text-xs text-slate-200/70">תקציר המקרה: {plan.case_summary}</p>
              </header>

              <section>
                <h3 className="text-sm font-semibold text-white">אסטרטגיה כללית</h3>
                <p className="mt-2 leading-relaxed">{plan.strategy}</p>
              </section>

              <section>
                <h3 className="text-sm font-semibold text-white">מוקדי חקירה</h3>
                <ul className="mt-3 flex flex-wrap gap-2 text-xs">
                  {plan.focus_areas.map((focus, index) => (
                    <li key={`${focus}-${index}`} className="rounded-full border border-white/20 bg-white/10 px-3 py-1 text-white">
                      {focus}
                    </li>
                  ))}
                </ul>
              </section>

              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-white">סטים של שאלות</h3>
                {plan.question_sets.map((set, index) => (
                  <div key={`${set.stage}-${index}`} className="rounded-3xl border border-white/10 bg-slate-950/40 p-4">
                    <h4 className="text-sm font-semibold text-white">{set.stage}</h4>
                    <ol className="mt-2 space-y-2 text-slate-100/90">
                      {set.questions.map((question, questionIndex) => (
                        <li key={`${question}-${questionIndex}`} className="list-decimal pr-4">
                          {question}
                        </li>
                      ))}
                    </ol>
                  </div>
                ))}
              </section>

              <section className="space-y-3">
                <h3 className="text-sm font-semibold text-white">בקרות וניהול סיכונים</h3>
                <ul className="space-y-2 text-xs text-slate-200/80">
                  {plan.risk_controls.map((item, index) => (
                    <li key={`${item}-${index}`}>{item}</li>
                  ))}
                </ul>
                {plan.quality_notes.length > 0 ? (
                  <div className="rounded-3xl border border-amber-200/60 bg-amber-500/10 p-4 text-xs text-amber-100">
                    <h4 className="font-semibold">הערות איכות</h4>
                    <ul className="mt-2 space-y-1">
                      {plan.quality_notes.map((note, index) => (
                        <li key={`${note}-${index}`}>{note}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </section>

              <section className="space-y-3">
                <h3 className="text-sm font-semibold text-white">הקשרים מהמסמכים שלך</h3>
                <ul className="space-y-3 text-sm">
                  {plan.contextual_references.map((reference, index) => (
                    <li key={`${reference.document_id}-${index}`} className="rounded-3xl border border-white/10 bg-white/5 p-4">
                      <div className="flex items-center justify-between text-[0.7rem] text-blue-200/80">
                        <span className="font-semibold text-white">{reference.filename}</span>
                        <span>ציון {reference.score.toFixed(2)}</span>
                      </div>
                      {reference.explanation ? (
                        <p className="mt-2 text-xs text-slate-200/80">{reference.explanation}</p>
                      ) : null}
                      <p className="mt-3 leading-relaxed text-slate-100/90">{reference.snippet}</p>
                    </li>
                  ))}
                </ul>
              </section>
            </article>
          )}
        </div>
      </section>
    </AppShell>
  );
}
