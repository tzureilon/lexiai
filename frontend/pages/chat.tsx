"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "../components/AppShell";
import { PageEmptyState, PageLoader } from "../components/PageState";
import { apiFetch } from "../lib/api";
import { useRequireAuth } from "../lib/auth";

interface ChatMessage {
  role: string;
  content: string;
  created_at: string;
}

interface ContextualReference {
  document_id: number;
  filename: string;
  snippet: string;
  score: number;
  explanation?: string;
}

export default function ChatPage() {
  const { session, loading } = useRequireAuth();
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [contextualReferences, setContextualReferences] = useState<ContextualReference[]>([]);

  const hasMessages = history.length > 0;

  const loadHistory = useCallback(async () => {
    if (!session) {
      return;
    }
    try {
      const data = await apiFetch<ChatMessage[]>(`/chat/history/${session.userId}?tenant_id=${session.tenantId}`);
      setHistory(data);
      setContextualReferences([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "שגיאה בטעינת היסטוריה");
    }
  }, [session]);

  useEffect(() => {
    if (!loading && session) {
      void loadHistory();
    }
  }, [loadHistory, loading, session]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!message.trim() || !session) return;
    setIsLoading(true);
    setError(null);
    try {
      const data = await apiFetch<{
        response: string;
        history: ChatMessage[];
        contextual_references: ContextualReference[];
      }>("/chat", {
        method: "POST",
        body: JSON.stringify({ tenant_id: session.tenantId, user_id: session.userId, message }),
      });
      setHistory(data.history);
      setContextualReferences(data.contextual_references ?? []);
      setMessage("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "שגיאה בשליחת הודעה");
    } finally {
      setIsLoading(false);
    }
  };

  const groupedMessages = useMemo(
    () => history.map((msg, idx) => ({ ...msg, id: `${msg.role}-${idx}` })),
    [history],
  );

  if (loading || !session) {
    return <PageLoader />;
  }

  const hero = (
    <div className="grid gap-4 md:grid-cols-3">
      {["היסטוריית שיחות", "מסמכים מוצלבים", "רמת ביטחון"].map((label, index) => (
        <div key={label} className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
          <p className="text-xs font-semibold text-blue-200/70">{label}</p>
          <p className="mt-2 text-2xl font-bold text-white">
            {index === 0 && history.length}
            {index === 1 && contextualReferences.length}
            {index === 2 && "Claude"}
          </p>
          <p className="mt-1 text-[0.7rem] text-slate-200/70">
            {index === 0 && "מספר הודעות שנשמרו בסשן הפעיל."}
            {index === 1 && "קטעי מסמכים שהוזכרו בתשובות האחרונות."}
            {index === 2 && "שכבת LLM פעילה עם מנגנוני נפילה מבוקרים."}
          </p>
        </div>
      ))}
    </div>
  );

  return (
    <AppShell
      title="חדר הפיקוד המשפטי"
      subtitle="שוחח עם העוזר המשפטי של LexiAI, הפק תשובות מבוססות מסמכים ונתיבי פעולה אופרטיביים לכל תרחיש."
      hero={hero}
    >
      <section className="grid gap-6 lg:grid-cols-[minmax(0,2.2fr)_minmax(280px,1fr)]">
        <div className="glass-panel flex h-full flex-col space-y-4">
          <div className="flex items-center justify-between text-xs text-slate-200/70">
            <span>היסטוריה חיה של הסשן</span>
            <button
              type="button"
              onClick={() => void loadHistory()}
              className="rounded-full border border-white/20 px-3 py-1 text-[0.7rem] font-semibold text-slate-200 transition hover:border-blue-200/60 hover:text-blue-100"
            >
              רענון
            </button>
          </div>
          <div className="flex-1 space-y-3 overflow-y-auto rounded-3xl bg-slate-950/40 p-4">
            {!hasMessages ? (
              <PageEmptyState
                title="לא נשלחו הודעות בסשן הנוכחי"
                description="שאל את LexiAI על קובצי הלקוח, על רציונל המודל או על פעולות הציות האחרונות, וקבל תשובה עם אסמכתאות."
              />
            ) : (
              groupedMessages.map((msg) => (
                <article key={msg.id} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <div className="flex items-center justify-between text-[0.7rem] font-semibold text-blue-200/80">
                    <span>{msg.role === "assistant" ? "LexiAI" : "משתמש"}</span>
                    <time className="text-[0.65rem] text-slate-300/70">{new Date(msg.created_at).toLocaleString()}</time>
                  </div>
                  <p className="mt-3 whitespace-pre-line text-sm leading-relaxed text-slate-100/90">{msg.content}</p>
                </article>
              ))
            )}
          </div>
          {error ? <p className="text-sm text-red-300">{error}</p> : null}
          <form onSubmit={handleSubmit} className="space-y-3 rounded-3xl border border-white/10 bg-white/5 p-4">
            <label className="text-xs font-semibold text-blue-200/80" htmlFor="message">
              ההודעה שלך
            </label>
            <textarea
              id="message"
              className="h-32 w-full resize-none rounded-2xl border border-white/20 bg-slate-950/60 p-3 text-sm text-white placeholder:text-slate-400 focus:border-blue-300 focus:outline-none"
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              placeholder="לדוגמה: מהן ההתחייבויות העיקריות בהסכם השכירות האחרון ואילו פעולות ציות מומלצות?"
            />
            <div className="flex flex-wrap justify-end gap-3 text-sm">
              <button
                type="button"
                className="rounded-full border border-white/20 px-4 py-2 font-semibold text-slate-200 transition hover:border-white/40 hover:bg-white/10"
                onClick={() => setMessage("")}
                disabled={!message}
              >
                ניקוי
              </button>
              <button
                type="submit"
                className="rounded-full bg-blue-500 px-6 py-2 font-semibold text-white shadow-[0_10px_30px_rgba(59,130,246,0.35)] transition hover:bg-blue-600"
                disabled={isLoading}
              >
                {isLoading ? "שולח..." : "שליחה"}
              </button>
            </div>
          </form>
        </div>
        <aside className="glass-panel flex h-full flex-col space-y-4">
          <div>
            <p className="text-sm font-semibold text-white">הקשר מהמסמכים</p>
            <p className="mt-1 text-xs text-slate-200/80">
              המנוע מזהה מסמכים רלוונטיים, מציין את ציון ההתאמה ומסביר כיצד הקטע חיזק את ההמלצה.
            </p>
          </div>
          {contextualReferences.length === 0 ? (
            <PageEmptyState
              title="טרם נמצאו אסמכתאות"
              description="לאחר שליחת שאלה שתואמת למסמכים שהועלו יוצגו כאן קטעים רלוונטיים וציון ההתאמה שלהם."
            />
          ) : (
            <ul className="space-y-3 overflow-y-auto pr-2 text-sm">
              {contextualReferences.map((reference) => (
                <li key={`${reference.document_id}-${reference.score}`} className="rounded-3xl border border-white/10 bg-white/5 p-4">
                  <div className="flex items-center justify-between text-[0.7rem] text-blue-200/80">
                    <span className="font-semibold text-white">{reference.filename}</span>
                    <span>ציון {reference.score.toFixed(2)}</span>
                  </div>
                  {reference.explanation ? (
                    <p className="mt-2 text-xs text-slate-200/70">{reference.explanation}</p>
                  ) : null}
                  <p className="mt-3 text-sm leading-relaxed text-slate-100/90">{reference.snippet}</p>
                </li>
              ))}
            </ul>
          )}
        </aside>
      </section>
    </AppShell>
  );
}
