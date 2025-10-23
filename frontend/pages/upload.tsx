"use client";

import { ChangeEvent, useCallback, useEffect, useState } from "react";

import { AppShell } from "../components/AppShell";
import { PageEmptyState, PageLoader } from "../components/PageState";
import { apiFetch } from "../lib/api";
import { useRequireAuth } from "../lib/auth";

interface DocumentSummary {
  id: number;
  filename: string;
  size: number;
  uploaded_at: string;
  latest_version: number;
  retention_policy: string;
  sensitivity: string;
  preview: string;
}

async function fileToBase64(file: File): Promise<string> {
  const reader = new FileReader();
  return await new Promise((resolve, reject) => {
    reader.onerror = () => reject(new Error("כשל בקריאת הקובץ"));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result === "string") {
        const base64 = result.split(",").pop() ?? "";
        resolve(base64);
      } else {
        reject(new Error("לא ניתן להמיר את הקובץ"));
      }
    };
    reader.readAsDataURL(file);
  });
}

export default function UploadPage() {
  const { session, loading } = useRequireAuth();
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [userId, setUserId] = useState("");
  const [retentionPolicy, setRetentionPolicy] = useState("standard");
  const [sensitivity, setSensitivity] = useState("internal");
  const [changeNote, setChangeNote] = useState("");

  const refreshDocuments = useCallback(async () => {
    if (!session) return;
    try {
      const data = await apiFetch<DocumentSummary[]>(`/documents?tenant_id=${session.tenantId}`, { method: "GET" });
      setDocuments(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "שגיאה בטעינת המסמכים");
    }
  }, [session]);

  useEffect(() => {
    if (!loading && session) {
      setUserId(session.userId);
      void refreshDocuments();
    }
  }, [loading, refreshDocuments, session]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setFile(event.target.files?.[0] ?? null);
    setStatus(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file || !session) return;
    setIsUploading(true);
    setStatus(null);
    setError(null);
    try {
      const base64 = await fileToBase64(file);
      const result = await apiFetch<DocumentSummary>("/upload", {
        method: "POST",
        body: JSON.stringify({
          tenant_id: session.tenantId,
          user_id: userId || session.userId,
          filename: file.name,
          content: base64,
          retention_policy: retentionPolicy,
          sensitivity,
          change_note: changeNote || null,
        }),
      });
      setStatus(`הקובץ ${result.filename} נשמר בהצלחה (${result.size} תווים).`);
      setFile(null);
      setChangeNote("");
      await refreshDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "שגיאה בהעלאת הקובץ");
    } finally {
      setIsUploading(false);
    }
  };

  if (loading || !session) {
    return <PageLoader message="טוען את סביבת ההעלאה המאובטחת..." />;
  }

  const hero = (
    <div className="grid gap-4 md:grid-cols-3">
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">מסמכים שהועלו</p>
        <p className="mt-2 text-2xl font-bold text-white">{documents.length}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">זמין לכלי החיפוש והציות שלך.</p>
      </div>
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">משתמש מעלה</p>
        <p className="mt-2 text-2xl font-bold text-white">{userId || session.userId}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">כל העלאה משויכת למשתמש לצורך ביקורת.</p>
      </div>
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">רמת רגישות ברירת מחדל</p>
        <p className="mt-2 text-2xl font-bold text-white">{sensitivity}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">ניתן לשנות לכל העלאה.</p>
      </div>
    </div>
  );

  return (
    <AppShell
      title="העלאת מסמכים מבוקרת"
      subtitle="שמור על תהליך מסודר של גרסאות, תיעוד ושיוך משתמשים לכל מסמך שנכנס לפלטפורמה."
      hero={hero}
    >
      <section className="grid gap-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <div className="glass-panel space-y-6">
          <header className="space-y-2 text-right">
            <p className="text-sm font-semibold text-white">פרטי ההעלאה</p>
            <p className="text-xs text-slate-200/80">
              הגדירו מדיניות שימור, סיווג רגישות והערות שינוי לפני שהמסמך נשלח לעיבוד ולניתוח ML.
            </p>
          </header>
          <div className="space-y-4">
            <div>
              <label className="text-xs font-semibold text-blue-200/70" htmlFor="file">
                בחרו קובץ
              </label>
              <input
                id="file"
                type="file"
                onChange={handleFileChange}
                className="mt-2 w-full rounded-2xl border border-dashed border-white/30 bg-slate-950/40 px-4 py-3 text-sm text-white focus:border-blue-300 focus:outline-none"
              />
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="userId">
                  מזהה משתמש
                </label>
                <input
                  id="userId"
                  value={userId}
                  onChange={(event) => setUserId(event.target.value)}
                  className="mt-2 w-full rounded-2xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white placeholder:text-slate-400 focus:border-blue-300 focus:outline-none"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="sensitivity">
                  רגישות המסמך
                </label>
                <select
                  id="sensitivity"
                  value={sensitivity}
                  onChange={(event) => setSensitivity(event.target.value)}
                  className="mt-2 w-full rounded-2xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
                >
                  <option value="internal">פנימי</option>
                  <option value="confidential">סודי</option>
                  <option value="restricted">גישה מוגבלת</option>
                </select>
              </div>
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="retention">
                  מדיניות שימור
                </label>
                <select
                  id="retention"
                  value={retentionPolicy}
                  onChange={(event) => setRetentionPolicy(event.target.value)}
                  className="mt-2 w-full rounded-2xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
                >
                  <option value="standard">ברירת מחדל</option>
                  <option value="long-term">ארוך טווח</option>
                  <option value="legal-hold">Legal Hold</option>
                  <option value="delete-on-request">מחיקה לפי בקשה</option>
                </select>
              </div>
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="note">
                  הערת שינוי
                </label>
                <input
                  id="note"
                  value={changeNote}
                  onChange={(event) => setChangeNote(event.target.value)}
                  placeholder="לדוגמה: נוספה חתימה מעודכנת"
                  className="mt-2 w-full rounded-2xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white placeholder:text-slate-400 focus:border-blue-300 focus:outline-none"
                />
              </div>
            </div>
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="text-xs text-slate-200/70">
              העלאות נשמרות עם חותמת זמן, מזהה משתמש וגרסה ראשונה במסד הנתונים הרב-דיירי.
            </div>
            <button
              onClick={() => void handleUpload()}
              disabled={!file || isUploading}
              className="rounded-full bg-blue-500 px-6 py-2 text-sm font-semibold text-white shadow-[0_10px_30px_rgba(59,130,246,0.35)] transition hover:bg-blue-600 disabled:cursor-not-allowed"
            >
              {isUploading ? "מעלה..." : "שמור מסמך"}
            </button>
          </div>
          {status ? <p className="text-sm text-emerald-300">{status}</p> : null}
          {error ? <p className="text-sm text-red-300">{error}</p> : null}
        </div>

        <aside className="glass-panel space-y-6">
          <header className="space-y-2 text-right">
            <p className="text-sm font-semibold text-white">מסמכים שהועלו לאחרונה</p>
            <p className="text-xs text-slate-200/80">
              המערכת מבצעת תמצות, ניתוח סיכונים ותיעוד רגולטורי מיד לאחר ההעלאה. ניתן לעבור למסך המסמכים להמשך העמקה.
            </p>
          </header>
          {documents.length === 0 ? (
            <PageEmptyState
              title="עוד לא הועלו מסמכים"
              description="ברגע שתעלה את המסמך הראשון הוא יופיע כאן עם תצוגה מקדימה מהירה, רמת רגישות ומדיניות שימור."
            />
          ) : (
            <ul className="space-y-3 text-sm">
              {documents.slice(0, 6).map((doc) => (
                <li key={doc.id} className="rounded-3xl border border-white/10 bg-white/5 p-4">
                  <div className="flex items-center justify-between text-[0.7rem] text-blue-200/80">
                    <span className="font-semibold text-white">{doc.filename}</span>
                    <time>{new Date(doc.uploaded_at).toLocaleString()}</time>
                  </div>
                  <p className="mt-2 text-xs text-slate-200/80">גרסה {doc.latest_version} · {doc.retention_policy} · {doc.sensitivity}</p>
                  <p className="mt-2 max-h-20 overflow-hidden text-sm text-slate-100/85">{doc.preview}</p>
                </li>
              ))}
            </ul>
          )}
          <div className="rounded-3xl border border-white/10 bg-slate-950/40 p-5 text-xs text-slate-200/80">
            <p className="text-sm font-semibold text-white">טיפים לעמידה במדיניות</p>
            <ul className="mt-2 space-y-1">
              <li>• הקפידו על שם קובץ המתאר את סוג המסמך והתיק.</li>
              <li>• הוסיפו הערת שינוי מפורטת לגרסאות המאוחרות יותר.</li>
              <li>• בדקו את רמות הרגישות לפני שיתוף עם צוותים נוספים.</li>
            </ul>
          </div>
        </aside>
      </section>
    </AppShell>
  );
}
