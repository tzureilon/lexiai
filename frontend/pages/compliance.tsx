"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import { AppShell } from "../components/AppShell";
import { PageEmptyState, PageLoader } from "../components/PageState";
import { apiFetch } from "../lib/api";
import { useRequireAuth } from "../lib/auth";

interface PrivacyRequest {
  id: number;
  user_id: string;
  request_type: string;
  resource_type: string;
  resource_id?: string | null;
  reason?: string | null;
  status: string;
  requested_at: string;
  resolved_at?: string | null;
  resolution_note?: string | null;
}

interface AuditLogEntry {
  id: number;
  user_id?: string | null;
  action: string;
  resource_type?: string | null;
  resource_id?: string | null;
  metadata?: Record<string, unknown> | null;
  created_at: string;
}

interface WorkflowTask {
  id: number;
  case_id: string;
  title: string;
  status: string;
  assignee?: string | null;
  due_date?: string | null;
  created_at: string;
  updated_at: string;
  tags: string[];
}

const statusLabels: Record<string, string> = {
  backlog: "ברשימת עבודה",
  in_progress: "בתהליך",
  blocked: "חסום",
  done: "הושלם",
};

export default function CompliancePage() {
  const { session, loading } = useRequireAuth();
  const [privacyRequests, setPrivacyRequests] = useState<PrivacyRequest[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditLogEntry[]>([]);
  const [workflowTasks, setWorkflowTasks] = useState<WorkflowTask[]>([]);
  const [privacyForm, setPrivacyForm] = useState({
    user_id: "",
    request_type: "access",
    resource_type: "document",
    resource_id: "",
    reason: "",
  });
  const [privacyStatus, setPrivacyStatus] = useState<string | null>(null);
  const [privacyError, setPrivacyError] = useState<string | null>(null);

  const [taskForm, setTaskForm] = useState({
    case_id: "",
    title: "",
    assignee: "",
    due_date: "",
    tags: "",
  });
  const [taskStatus, setTaskStatus] = useState<string | null>(null);
  const [taskError, setTaskError] = useState<string | null>(null);

  const [auditFilter, setAuditFilter] = useState({ user_id: "", action: "" });

  const refreshPrivacy = useCallback(async () => {
    if (!session) return;
    const data = await apiFetch<PrivacyRequest[]>(`/privacy/requests?tenant_id=${session.tenantId}`);
    setPrivacyRequests(data);
  }, [session]);

  const refreshAudit = useCallback(async () => {
    if (!session) return;
    const params = new URLSearchParams();
    params.set("tenant_id", session.tenantId);
    if (auditFilter.user_id.trim()) params.set("user_id", auditFilter.user_id.trim());
    if (auditFilter.action.trim()) params.set("action", auditFilter.action.trim());
    const query = params.toString();
    const url = `/audit?${query}`;
    const data = await apiFetch<AuditLogEntry[]>(url);
    setAuditLogs(data);
  }, [auditFilter.action, auditFilter.user_id, session]);

  const refreshTasks = useCallback(async () => {
    if (!session) return;
    const data = await apiFetch<WorkflowTask[]>(`/workflows/tasks?tenant_id=${session.tenantId}`);
    setWorkflowTasks(data);
  }, [session]);

  useEffect(() => {
    if (!loading && session) {
      setPrivacyForm((current) => ({ ...current, user_id: session.userId }));
      void refreshPrivacy();
      void refreshAudit();
      void refreshTasks();
    }
  }, [loading, refreshAudit, refreshPrivacy, refreshTasks, session]);

  const handlePrivacySubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setPrivacyStatus(null);
    setPrivacyError(null);
    try {
      await apiFetch<PrivacyRequest>("/privacy/requests", {
        method: "POST",
        body: JSON.stringify({
          tenant_id: session.tenantId,
          user_id: privacyForm.user_id,
          request_type: privacyForm.request_type,
          resource_type: privacyForm.resource_type,
          resource_id: privacyForm.resource_id || null,
          reason: privacyForm.reason || null,
        }),
      });
      setPrivacyStatus("הבקשה נשמרה ונכנסה לתהליך טיפול.");
      setPrivacyForm({ ...privacyForm, resource_id: "", reason: "" });
      await refreshPrivacy();
      await refreshAudit();
    } catch (err) {
      setPrivacyError(err instanceof Error ? err.message : "לא ניתן להגיש בקשה חדשה");
    }
  };

  const handleTaskSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setTaskStatus(null);
    setTaskError(null);
    try {
      await apiFetch<WorkflowTask>("/workflows/tasks", {
        method: "POST",
        body: JSON.stringify({
          tenant_id: session.tenantId,
          case_id: taskForm.case_id,
          title: taskForm.title,
          assignee: taskForm.assignee || null,
          due_date: taskForm.due_date ? new Date(taskForm.due_date).toISOString() : null,
          tags: taskForm.tags
            .split(",")
            .map((tag) => tag.trim())
            .filter(Boolean),
        }),
      });
      setTaskStatus("משימה נוצרה בהצלחה.");
      setTaskForm({ case_id: "", title: "", assignee: "", due_date: "", tags: "" });
      await refreshTasks();
      await refreshAudit();
    } catch (err) {
      setTaskError(err instanceof Error ? err.message : "יצירת המשימה נכשלה");
    }
  };

  const handleTaskStatusChange = async (task: WorkflowTask, status: string) => {
    if (!session) return;
    await apiFetch<WorkflowTask>(`/workflows/tasks/${task.id}`, {
      method: "PATCH",
      body: JSON.stringify({ tenant_id: session.tenantId, status }),
    });
    await refreshTasks();
    await refreshAudit();
  };

  const sortedAudit = useMemo(
    () =>
      [...auditLogs].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()),
    [auditLogs],
  );

  if (loading || !session) {
    return <PageLoader message="טוען נתוני ציות מאובטחים..." />;
  }

  const hero = (
    <div className="grid gap-4 md:grid-cols-3">
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">בקשות פרטיות פעילות</p>
        <p className="mt-2 text-2xl font-bold text-white">{privacyRequests.filter((request) => request.status !== "done").length}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">מעקב אחר בקשות המחכה לטיפול.</p>
      </div>
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">משימות תפעול</p>
        <p className="mt-2 text-2xl font-bold text-white">{workflowTasks.length}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">מספר משימות הצוות לציות ופעילות.</p>
      </div>
      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 text-right">
        <p className="text-xs font-semibold text-blue-200/70">יומני ביקורת</p>
        <p className="mt-2 text-2xl font-bold text-white">{auditLogs.length}</p>
        <p className="mt-1 text-[0.7rem] text-slate-200/70">אירועים מתועדים ב-24 השעות האחרונות.</p>
      </div>
    </div>
  );

  return (
    <AppShell
      title="קונסול הציות והפעילות"
      subtitle="נהל בקשות פרטיות, זרימות עבודה ויומני ביקורת במרכז אחד עם שקיפות מלאה על כל פעולה."
      hero={hero}
    >
      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1.8fr)]">
        <div className="space-y-6">
          <div className="glass-panel space-y-5">
            <header className="space-y-2 text-right">
              <p className="text-sm font-semibold text-white">בקשות פרטיות</p>
              <p className="text-xs text-slate-200/80">רישום ומעקב אחר בקשות גישה, מחיקה ותיקון של נתונים רגישים.</p>
            </header>
            <form onSubmit={handlePrivacySubmit} className="grid gap-3 md:grid-cols-2">
              <div className="md:col-span-2">
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="privacy-user">
                  מזהה משתמש
                </label>
                <input
                  id="privacy-user"
                  value={privacyForm.user_id}
                  onChange={(event) => setPrivacyForm({ ...privacyForm, user_id: event.target.value })}
                  className="mt-2 w-full rounded-3xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="privacy-type">
                  סוג בקשה
                </label>
                <select
                  id="privacy-type"
                  value={privacyForm.request_type}
                  onChange={(event) => setPrivacyForm({ ...privacyForm, request_type: event.target.value })}
                  className="mt-2 w-full rounded-3xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
                >
                  <option value="access">בקשת גישה</option>
                  <option value="delete">בקשת מחיקה</option>
                  <option value="rectify">בקשת תיקון</option>
                </select>
              </div>
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="privacy-resource">
                  סוג משאב
                </label>
                <select
                  id="privacy-resource"
                  value={privacyForm.resource_type}
                  onChange={(event) => setPrivacyForm({ ...privacyForm, resource_type: event.target.value })}
                  className="mt-2 w-full rounded-3xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
                >
                  <option value="document">מסמך</option>
                  <option value="conversation">שיחה</option>
                  <option value="task">משימה</option>
                </select>
              </div>
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="privacy-resource-id">
                  מזהה משאב
                </label>
                <input
                  id="privacy-resource-id"
                  value={privacyForm.resource_id}
                  onChange={(event) => setPrivacyForm({ ...privacyForm, resource_id: event.target.value })}
                  className="mt-2 w-full rounded-3xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
                />
              </div>
              <div className="md:col-span-2">
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="privacy-reason">
                  נימוק
                </label>
                <textarea
                  id="privacy-reason"
                  value={privacyForm.reason}
                  onChange={(event) => setPrivacyForm({ ...privacyForm, reason: event.target.value })}
                  className="mt-2 h-20 w-full rounded-3xl border border-white/20 bg-slate-950/50 p-3 text-sm text-white focus:border-blue-300 focus:outline-none"
                  placeholder="תאר את הנימוק או בקשת הלקוח"
                />
              </div>
              {privacyError ? <p className="md:col-span-2 text-sm text-red-300">{privacyError}</p> : null}
              {privacyStatus ? <p className="md:col-span-2 text-sm text-emerald-300">{privacyStatus}</p> : null}
              <div className="md:col-span-2 flex justify-end">
                <button
                  type="submit"
                  className="rounded-full bg-blue-500 px-6 py-2 text-sm font-semibold text-white transition hover:bg-blue-600"
                >
                  הגשת בקשה
                </button>
              </div>
            </form>
            <div className="rounded-3xl border border-white/10 bg-slate-950/40 p-4 text-xs text-slate-200/80">
              <h3 className="text-sm font-semibold text-white">בקשות אחרונות</h3>
              {privacyRequests.length === 0 ? (
                <p className="mt-2">אין בקשות פעילות כרגע.</p>
              ) : (
                <ul className="mt-3 space-y-2">
                  {privacyRequests.slice(0, 6).map((request) => (
                    <li key={request.id} className="rounded-2xl border border-white/10 bg-white/5 p-3">
                      <div className="flex items-center justify-between text-[0.7rem] text-blue-200/80">
                        <span>{request.request_type} · {request.resource_type}</span>
                        <span>{statusLabels[request.status] ?? request.status}</span>
                      </div>
                      <p className="mt-1 text-xs text-slate-200/70">מבקש: {request.user_id}</p>
                      <p className="mt-1 text-xs text-slate-200/60">{new Date(request.requested_at).toLocaleString()}</p>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          <div className="glass-panel space-y-5">
            <header className="space-y-2 text-right">
              <p className="text-sm font-semibold text-white">משימות תפעול וציות</p>
              <p className="text-xs text-slate-200/80">צרו משימות חדשות, הקצו אחראים ועדכנו סטטוסים בזמן אמת.</p>
            </header>
            <form onSubmit={handleTaskSubmit} className="grid gap-3 md:grid-cols-2">
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="task-case">מזהה תיק</label>
                <input
                  id="task-case"
                  value={taskForm.case_id}
                  onChange={(event) => setTaskForm({ ...taskForm, case_id: event.target.value })}
                  className="mt-2 w-full rounded-3xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="task-title">כותרת המשימה</label>
                <input
                  id="task-title"
                  value={taskForm.title}
                  onChange={(event) => setTaskForm({ ...taskForm, title: event.target.value })}
                  className="mt-2 w-full rounded-3xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="task-assignee">אחראי</label>
                <input
                  id="task-assignee"
                  value={taskForm.assignee}
                  onChange={(event) => setTaskForm({ ...taskForm, assignee: event.target.value })}
                  className="mt-2 w-full rounded-3xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="task-due">תאריך יעד</label>
                <input
                  id="task-due"
                  type="date"
                  value={taskForm.due_date}
                  onChange={(event) => setTaskForm({ ...taskForm, due_date: event.target.value })}
                  className="mt-2 w-full rounded-3xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
                />
              </div>
              <div className="md:col-span-2">
                <label className="text-xs font-semibold text-blue-200/70" htmlFor="task-tags">תגיות</label>
                <input
                  id="task-tags"
                  value={taskForm.tags}
                  onChange={(event) => setTaskForm({ ...taskForm, tags: event.target.value })}
                  placeholder="מופרד בפסיקים: GDPR, SOC2, חקירת עד"
                  className="mt-2 w-full rounded-3xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
                />
              </div>
              {taskError ? <p className="md:col-span-2 text-sm text-red-300">{taskError}</p> : null}
              {taskStatus ? <p className="md:col-span-2 text-sm text-emerald-300">{taskStatus}</p> : null}
              <div className="md:col-span-2 flex justify-end">
                <button
                  type="submit"
                  className="rounded-full bg-blue-500 px-6 py-2 text-sm font-semibold text-white transition hover:bg-blue-600"
                >
                  צור משימה
                </button>
              </div>
            </form>
            <div className="rounded-3xl border border-white/10 bg-slate-950/40 p-4 text-xs text-slate-200/80">
              <h3 className="text-sm font-semibold text-white">משימות פעילות</h3>
              {workflowTasks.length === 0 ? (
                <p className="mt-2">אין משימות פעילות.</p>
              ) : (
                <ul className="mt-3 space-y-2">
                  {workflowTasks.map((task) => (
                    <li key={task.id} className="rounded-2xl border border-white/10 bg-white/5 p-3">
                      <div className="flex items-center justify-between text-[0.7rem] text-blue-200/80">
                        <span className="font-semibold text-white">{task.title}</span>
                        <select
                          value={task.status}
                          onChange={(event) => void handleTaskStatusChange(task, event.target.value)}
                          className="rounded-full border border-white/20 bg-slate-950/60 px-3 py-1 text-xs text-white focus:border-blue-300 focus:outline-none"
                        >
                          {Object.entries(statusLabels).map(([value, label]) => (
                            <option key={value} value={value}>
                              {label}
                            </option>
                          ))}
                        </select>
                      </div>
                      <p className="mt-1 text-xs text-slate-200/70">תיק: {task.case_id}</p>
                      <p className="mt-1 text-xs text-slate-200/60">אחראי: {task.assignee ?? "לא מוקצה"}</p>
                      <p className="mt-1 text-xs text-slate-200/60">
                        יעד: {task.due_date ? new Date(task.due_date).toLocaleDateString() : "ללא"}
                      </p>
                      <div className="mt-2 flex flex-wrap gap-2 text-[0.65rem]">
                        {task.tags.map((tag) => (
                          <span key={tag} className="rounded-full border border-white/20 bg-white/10 px-2 py-1 text-white">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>

        <div className="glass-panel space-y-5">
          <header className="space-y-2 text-right">
            <p className="text-sm font-semibold text-white">יומן ביקורת</p>
            <p className="text-xs text-slate-200/80">חפשו אירועים לפי משתמש או פעולה כדי לעקוב אחר פעילות רגישה.</p>
          </header>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void refreshAudit();
            }}
            className="grid gap-3 md:grid-cols-2"
          >
            <div>
              <label className="text-xs font-semibold text-blue-200/70" htmlFor="audit-user">משתמש</label>
              <input
                id="audit-user"
                value={auditFilter.user_id}
                onChange={(event) => setAuditFilter({ ...auditFilter, user_id: event.target.value })}
                className="mt-2 w-full rounded-3xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
              />
            </div>
            <div>
              <label className="text-xs font-semibold text-blue-200/70" htmlFor="audit-action">פעולה</label>
              <input
                id="audit-action"
                value={auditFilter.action}
                onChange={(event) => setAuditFilter({ ...auditFilter, action: event.target.value })}
                className="mt-2 w-full rounded-3xl border border-white/20 bg-slate-950/50 px-3 py-2 text-sm text-white focus:border-blue-300 focus:outline-none"
              />
            </div>
            <div className="md:col-span-2 flex justify-end">
              <button
                type="submit"
                className="rounded-full bg-blue-500 px-6 py-2 text-sm font-semibold text-white transition hover:bg-blue-600"
              >
                סינון
              </button>
            </div>
          </form>
          {sortedAudit.length === 0 ? (
            <PageEmptyState
              title="אין אירועים להצגה"
              description="התוצאות יתעדכנו אוטומטית לאחר פעולות חדשות במערכת או לאחר שינוי הפילטרים."
            />
          ) : (
            <ul className="space-y-3 text-xs text-slate-200/80">
              {sortedAudit.slice(0, 12).map((entry) => (
                <li key={entry.id} className="rounded-3xl border border-white/10 bg-white/5 p-4">
                  <div className="flex items-center justify-between text-[0.7rem] text-blue-200/80">
                    <span>{entry.action}</span>
                    <time>{new Date(entry.created_at).toLocaleString()}</time>
                  </div>
                  <p className="mt-1 text-xs text-slate-200/70">משתמש: {entry.user_id ?? "מערכת"}</p>
                  {entry.resource_type ? (
                    <p className="mt-1 text-xs text-slate-200/70">
                      משאב: {entry.resource_type} · {entry.resource_id ?? "—"}
                    </p>
                  ) : null}
                  {entry.metadata ? (
                    <pre className="mt-2 whitespace-pre-wrap rounded-2xl border border-white/10 bg-slate-950/40 p-2 text-[0.65rem] text-slate-200/70">
                      {JSON.stringify(entry.metadata, null, 2)}
                    </pre>
                  ) : null}
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </AppShell>
  );
}
