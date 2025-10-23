"use client";

import { ChangeEvent, FormEvent, useCallback, useEffect, useState } from "react";

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
  owner_id?: string | null;
  preview: string;
}

interface DocumentInsights {
  key_points: string[];
  obligations: string[];
  risks: string[];
  deadlines: string[];
  recommended_actions: string[];
  rationale: string[];
  confidence_score: number;
  model_version?: string;
}

interface DocumentVersion {
  version: number;
  created_at: string;
  checksum: string;
  created_by?: string | null;
  change_note?: string | null;
}

interface DocumentDetails extends DocumentSummary {
  content: string;
  insights: DocumentInsights;
  versions: DocumentVersion[];
}

interface DocumentSearchResult {
  document_id: number;
  filename: string;
  snippet: string;
  score: number;
}

export default function DocumentsPage() {
  const { session, loading } = useRequireAuth();
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selected, setSelected] = useState<DocumentDetails | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<DocumentSearchResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [metadataRetention, setMetadataRetention] = useState("standard");
  const [metadataSensitivity, setMetadataSensitivity] = useState("internal");
  const [metadataUser, setMetadataUser] = useState("");
  const [metadataStatus, setMetadataStatus] = useState<string | null>(null);
  const [metadataError, setMetadataError] = useState<string | null>(null);
  const [isUpdatingMetadata, setIsUpdatingMetadata] = useState(false);
  const [versionFile, setVersionFile] = useState<File | null>(null);
  const [versionChangeNote, setVersionChangeNote] = useState("");
  const [versionStatus, setVersionStatus] = useState<string | null>(null);
  const [versionError, setVersionError] = useState<string | null>(null);
  const [isUploadingVersion, setIsUploadingVersion] = useState(false);

  const loadDocuments = useCallback(async () => {
    if (!session) return;
    try {
      const data = await apiFetch<DocumentSummary[]>(`/documents?tenant_id=${session.tenantId}`);
      setDocuments(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "שגיאה בטעינת המסמכים");
    }
  }, [session]);

  useEffect(() => {
    if (!loading && session) {
      setMetadataUser(session.userId);
      void loadDocuments();
    }
  }, [loadDocuments, loading, session]);

  const loadDocument = useCallback(
    async (id: number) => {
      if (!session) return;
      setIsLoading(true);
      setError(null);
      try {
        const data = await apiFetch<DocumentDetails>(
          `/documents/${id}?tenant_id=${session.tenantId}&requester_id=${session.userId}`,
        );
        setSelected(data);
        setMetadataRetention(data.retention_policy);
        setMetadataSensitivity(data.sensitivity);
        setMetadataUser(data.owner_id ?? session.userId);
        setMetadataStatus(null);
        setMetadataError(null);
        setVersionFile(null);
        setVersionChangeNote("");
        setVersionStatus(null);
        setVersionError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "שגיאה בפתיחת המסמך");
      } finally {
        setIsLoading(false);
      }
    },
    [session],
  );

  const handleSearch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!session) return;
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }
    setIsSearching(true);
    setError(null);
    try {
      const data = await apiFetch<DocumentSearchResult[]>(
        `/documents/search?tenant_id=${session.tenantId}&query=${encodeURIComponent(searchQuery)}&limit=6`,
      );
      setSearchResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "שגיאה בביצוע החיפוש");
    } finally {
      setIsSearching(false);
    }
  };

  const handleMetadataUpdate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selected || !session) return;
    setIsUpdatingMetadata(true);
    setMetadataStatus(null);
    setMetadataError(null);
    try {
      const updated = await apiFetch<DocumentDetails>(`/documents/${selected.id}`, {
        method: "PATCH",
        body: JSON.stringify({
          tenant_id: session.tenantId,
          user_id: metadataUser || session.userId,
          retention_policy: metadataRetention,
          sensitivity: metadataSensitivity,
        }),
      });
      setSelected(updated);
      setMetadataStatus("מדיניות השימור והרגישות עודכנו בהצלחה.");
      await loadDocuments();
    } catch (err) {
      setMetadataError(err instanceof Error ? err.message : "עדכון המדיניות נכשל");
    } finally {
      setIsUpdatingMetadata(false);
    }
  };

  const handleVersionFile = (event: ChangeEvent<HTMLInputElement>) => {
    setVersionFile(event.target.files?.[0] ?? null);
    setVersionStatus(null);
    setVersionError(null);
  };

  const fileToBase64 = async (file: File): Promise<string> => {
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
  };

  const handleVersionUpload = async () => {
    if (!selected || !versionFile || !session) return;
    setIsUploadingVersion(true);
    setVersionStatus(null);
    setVersionError(null);
    try {
      const base64 = await fileToBase64(versionFile);
      const updated = await apiFetch<DocumentDetails>(`/documents/${selected.id}/versions`, {
        method: "POST",
        body: JSON.stringify({
          tenant_id: session.tenantId,
          user_id: metadataUser || session.userId,
          content: base64,
          change_note: versionChangeNote || null,
        }),
      });
      setSelected(updated);
      setVersionStatus("הגרסה החדשה נשמרה ונוספה להיסטוריה.");
      await loadDocuments();
    } catch (err) {
      setVersionError(err instanceof Error ? err.message : "העלאת הגרסה נכשלה");
    } finally {
      setIsUploadingVersion(false);
    }
  };

  if (loading || !session) {
    return <PageLoader />;
  }

  const hero = (
    <div className="grid gap-4 md:grid-cols-3">
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">מסמכים במערכת</p>
        <p className="mt-2 text-2xl font-bold text-white">{documents.length}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">כולל גרסאות ותיוגי רגישות מעודכנים.</p>
      </div>
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">מדיניות שימור פעילה</p>
        <p className="mt-2 text-2xl font-bold text-white">{metadataRetention}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">מועדכן לפי המסמך האחרון שנפתח.</p>
      </div>
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">גרסה נוכחית</p>
        <p className="mt-2 text-2xl font-bold text-white">{selected?.latest_version ?? "—"}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">כולל נתוני checksum ותיעוד מלא של יומן הפעילות.</p>
      </div>
    </div>
  );

  return (
    <AppShell
      title="ספריית המסמכים החכמה"
      subtitle="נהל גרסאות, רגולציה ותובנות ML לכל מסמך – החל מהעלאה ועד למחיקה מבוקרת."
      hero={hero}
    >
      <section className="grid gap-6 xl:grid-cols-[minmax(280px,0.9fr)_minmax(0,2.1fr)]">
        <div className="glass-panel space-y-6">
          <div>
            <p className="text-sm font-semibold text-white">חיפוש במסמכים</p>
            <p className="mt-1 text-xs text-slate-200/80">בצעו חיפוש סמנטי במסמכים כדי לאתר קטעים רלוונטיים בזמן אמת.</p>
          </div>
          <form onSubmit={handleSearch} className="space-y-3">
            <div className="rounded-2xl border border-white/15 bg-white/5 p-3">
              <input
                className="w-full rounded-xl border border-white/10 bg-slate-950/60 px-3 py-2 text-sm text-white placeholder:text-slate-400 focus:border-blue-300 focus:outline-none"
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="לדוגמה: התחייבויות סודיות בהסכם רכישה"
              />
            </div>
            <button
              type="submit"
              className="w-full rounded-full bg-blue-500 px-5 py-2 text-sm font-semibold text-white transition hover:bg-blue-600"
              disabled={isSearching}
            >
              {isSearching ? "מחפש..." : "בצע חיפוש"}
            </button>
          </form>
          {searchResults.length > 0 ? (
            <div className="space-y-3">
              <p className="text-xs font-semibold text-blue-200/70">תוצאות אחרונות</p>
              <ul className="space-y-3 text-sm">
                {searchResults.map((result) => (
                  <li key={`${result.document_id}-${result.score}`} className="rounded-3xl border border-white/10 bg-white/5 p-4">
                    <div className="flex items-center justify-between text-[0.7rem] text-blue-200/80">
                      <span className="font-semibold text-white">{result.filename}</span>
                      <span>ציון {result.score.toFixed(2)}</span>
                    </div>
                    <p className="mt-2 text-sm text-slate-200/80">{result.snippet}</p>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          <div>
            <p className="text-sm font-semibold text-white">כל המסמכים</p>
            <p className="mt-1 text-xs text-slate-200/70">בחרו מסמך כדי לראות תובנות, גרסאות ופרטי רגולציה.</p>
          </div>
          <div className="max-h-[26rem] space-y-3 overflow-y-auto pr-2">
            {documents.map((doc) => {
              const isActive = selected?.id === doc.id;
              return (
                <button
                  key={doc.id}
                  type="button"
                  onClick={() => void loadDocument(doc.id)}
                  className={`w-full rounded-3xl border px-4 py-3 text-right transition ${
                    isActive
                      ? "border-blue-300/60 bg-blue-500/20 text-white"
                      : "border-white/10 bg-white/5 text-slate-200 hover:border-white/25 hover:bg-white/10"
                  }`}
                >
                  <div className="flex items-center justify-between text-xs font-semibold">
                    <span>{doc.filename}</span>
                    <span>גרסה {doc.latest_version}</span>
                  </div>
                  <p className="mt-1 text-[0.7rem] text-slate-200/70">גודל {Math.round(doc.size / 1024)}KB · {new Date(doc.uploaded_at).toLocaleDateString()}</p>
                  <p className="mt-2 max-h-16 overflow-hidden text-[0.75rem] text-slate-100/80">{doc.preview}</p>
                </button>
              );
            })}
          </div>
        </div>

        <div className="space-y-6">
          <div className="glass-panel">
            {selected ? (
              <div className="space-y-6">
                <header className="flex flex-col gap-2 text-right">
                  <p className="text-xs font-semibold uppercase tracking-[0.3em] text-blue-200/70">מסמך נבחר</p>
                  <h2 className="text-2xl font-bold text-white">{selected.filename}</h2>
                  <p className="text-xs text-slate-200/80">
                    רמת רגישות {selected.sensitivity} · מדיניות שימור {selected.retention_policy} · בבעלות {selected.owner_id ?? "לא משויך"}
                  </p>
                </header>
                <section className="rounded-3xl border border-white/10 bg-white/5 p-5">
                  <p className="text-sm font-semibold text-white">תובנות ומדדים</p>
                  <p className="mt-1 text-xs text-slate-200/80">ציון הביטחון של המודל: {(selected.insights.confidence_score * 100).toFixed(0)}%.</p>
                  <div className="mt-4 grid gap-4 md:grid-cols-2">
                    {(
                      [
                        { label: "נקודות מפתח", items: selected.insights.key_points },
                        { label: "התחייבויות", items: selected.insights.obligations },
                        { label: "סיכונים", items: selected.insights.risks },
                        { label: "דדליינים", items: selected.insights.deadlines },
                      ] as const
                    ).map((group) => (
                      <div key={group.label} className="rounded-2xl border border-white/10 bg-slate-950/50 p-4 text-sm">
                        <p className="text-xs font-semibold text-blue-200/70">{group.label}</p>
                        <ul className="mt-2 space-y-2 text-slate-100/90">
                          {group.items.length > 0 ? (
                            group.items.map((item, index) => <li key={`${group.label}-${index}`}>{item}</li>)
                          ) : (
                            <li className="text-xs text-slate-400">אין נתונים להצגה.</li>
                          )}
                        </ul>
                      </div>
                    ))}
                  </div>
                  <div className="mt-4 rounded-2xl border border-white/10 bg-slate-950/40 p-4 text-sm">
                    <p className="text-xs font-semibold text-blue-200/70">המלצות להמשך</p>
                    <ul className="mt-2 space-y-2">
                      {selected.insights.recommended_actions.map((item, index) => (
                        <li key={`action-${index}`} className="text-slate-100/90">
                          {item}
                        </li>
                      ))}
                    </ul>
                    <p className="mt-4 text-xs text-slate-200/70">רציונל המודל:</p>
                    <ul className="mt-2 space-y-1 text-xs text-slate-200/70">
                      {selected.insights.rationale.map((item, index) => (
                        <li key={`rationale-${index}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </section>
                <section className="rounded-3xl border border-white/10 bg-white/5 p-5">
                  <form onSubmit={handleMetadataUpdate} className="grid gap-4 md:grid-cols-3">
                    <div className="md:col-span-1">
                      <label className="text-xs font-semibold text-blue-200/70" htmlFor="owner">
                        בעל המסמך
                      </label>
                      <input
                        id="owner"
                        className="mt-2 w-full rounded-2xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white placeholder:text-slate-500 focus:border-blue-300 focus:outline-none"
                        value={metadataUser}
                        onChange={(event) => setMetadataUser(event.target.value)}
                      />
                    </div>
                    <div>
                      <label className="text-xs font-semibold text-blue-200/70" htmlFor="retention">
                        מדיניות שימור
                      </label>
                      <select
                        id="retention"
                        className="mt-2 w-full rounded-2xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
                        value={metadataRetention}
                        onChange={(event) => setMetadataRetention(event.target.value)}
                      >
                        <option value="standard">סטנדרט</option>
                        <option value="long-term">ארוך טווח</option>
                        <option value="legal-hold">Legal Hold</option>
                        <option value="delete-on-request">מחיקה לפי בקשה</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs font-semibold text-blue-200/70" htmlFor="sensitivity">
                        רמת רגישות
                      </label>
                      <select
                        id="sensitivity"
                        className="mt-2 w-full rounded-2xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
                        value={metadataSensitivity}
                        onChange={(event) => setMetadataSensitivity(event.target.value)}
                      >
                        <option value="internal">פנימי</option>
                        <option value="confidential">סודי</option>
                        <option value="restricted">גישה מוגבלת</option>
                      </select>
                    </div>
                    {metadataError ? <p className="md:col-span-3 text-sm text-red-300">{metadataError}</p> : null}
                    {metadataStatus ? <p className="md:col-span-3 text-sm text-emerald-300">{metadataStatus}</p> : null}
                    <div className="md:col-span-3 flex justify-end">
                      <button
                        type="submit"
                        className="rounded-full bg-blue-500 px-6 py-2 text-sm font-semibold text-white transition hover:bg-blue-600"
                        disabled={isUpdatingMetadata}
                      >
                        {isUpdatingMetadata ? "מעדכן..." : "עדכון מדיניות"}
                      </button>
                    </div>
                  </form>
                </section>
                <section className="rounded-3xl border border-white/10 bg-white/5 p-5">
                  <div className="flex flex-col gap-3 md:flex-row md:items-end">
                    <div className="flex-1">
                      <label className="text-xs font-semibold text-blue-200/70" htmlFor="versionFile">
                        העלאת גרסה חדשה
                      </label>
                      <input
                        id="versionFile"
                        type="file"
                        onChange={handleVersionFile}
                        className="mt-2 w-full rounded-2xl border border-dashed border-white/30 bg-slate-950/40 px-3 py-3 text-sm text-white focus:border-blue-300 focus:outline-none"
                      />
                    </div>
                    <div className="flex-1">
                      <label className="text-xs font-semibold text-blue-200/70" htmlFor="changeNote">
                        הערת שינוי (אופציונלי)
                      </label>
                      <input
                        id="changeNote"
                        className="mt-2 w-full rounded-2xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white placeholder:text-slate-500 focus:border-blue-300 focus:outline-none"
                        value={versionChangeNote}
                        onChange={(event) => setVersionChangeNote(event.target.value)}
                      />
                    </div>
                    <button
                      type="button"
                      onClick={() => void handleVersionUpload()}
                      className="rounded-full bg-blue-500 px-6 py-3 text-sm font-semibold text-white transition hover:bg-blue-600"
                      disabled={!versionFile || isUploadingVersion}
                    >
                      {isUploadingVersion ? "מעלה..." : "שמור גרסה"}
                    </button>
                  </div>
                  {versionError ? <p className="mt-3 text-sm text-red-300">{versionError}</p> : null}
                  {versionStatus ? <p className="mt-3 text-sm text-emerald-300">{versionStatus}</p> : null}
                  <div className="mt-6">
                    <p className="text-xs font-semibold text-blue-200/70">היסטוריית גרסאות</p>
                    <ul className="mt-3 grid gap-3 md:grid-cols-2">
                      {selected.versions.map((version) => (
                        <li key={version.version} className="rounded-2xl border border-white/10 bg-slate-950/40 p-4 text-xs text-slate-200/80">
                          <p className="text-sm font-semibold text-white">גרסה {version.version}</p>
                          <p className="mt-1">נוצר ב-{new Date(version.created_at).toLocaleString()}</p>
                          <p className="mt-1">Checksum: {version.checksum}</p>
                          {version.change_note ? <p className="mt-1 text-slate-200">{version.change_note}</p> : null}
                          {version.created_by ? <p className="mt-1 text-slate-200/70">על ידי {version.created_by}</p> : null}
                        </li>
                      ))}
                    </ul>
                  </div>
                </section>
              </div>
            ) : (
              <PageEmptyState
                title="בחרו מסמך מתוך הספרייה"
                description="המסמך הנבחר יופיע כאן עם תובנות ML, היסטוריית גרסאות ויכולות ניהול מדיניות."
              />
            )}
          </div>
          {error ? <p className="text-sm text-red-300">{error}</p> : null}
          {isLoading ? <p className="text-sm text-blue-200/80">טוען את פרטי המסמך...</p> : null}
        </div>
      </section>
    </AppShell>
  );
}
