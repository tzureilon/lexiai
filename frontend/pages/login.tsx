"use client";

import { FormEvent, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/router";

import { apiFetch } from "../lib/api";
import { useAuth } from "../lib/auth";

interface LoginPayload {
  token: string;
  token_id: string;
  expires_at?: string | null;
  roles: string[];
}

export default function LoginPage() {
  const router = useRouter();
  const { login, session } = useAuth();
  const [tenantId, setTenantId] = useState("demo-tenant");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (session) {
      void router.replace("/");
    }
  }, [router, session]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const response = await apiFetch<LoginPayload>("/auth/login", {
        method: "POST",
        body: JSON.stringify({ tenant_id: tenantId, email, password }),
      });
      login(response.token);
      const next = typeof router.query.next === "string" ? router.query.next : "/";
      await router.replace(next || "/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "ההתחברות נכשלה");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden px-4 py-12 text-right">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute top-[-12rem] left-1/2 h-[28rem] w-[28rem] -translate-x-1/2 rounded-full bg-blue-500/30 blur-[160px]" />
        <div className="absolute bottom-[-14rem] right-1/3 h-[32rem] w-[32rem] rounded-full bg-purple-500/25 blur-[160px]" />
      </div>
      <div className="relative grid w-full max-w-5xl gap-10 rounded-[2.5rem] border border-white/10 bg-white/5 p-10 shadow-[0_40px_120px_rgba(15,23,42,0.55)] backdrop-blur-xl lg:grid-cols-[1.3fr_1fr]">
        <section className="flex flex-col justify-between space-y-8">
          <div className="space-y-4">
            <p className="text-xs font-semibold uppercase tracking-[0.4em] text-blue-200/70">LexiAI Access</p>
            <h1 className="text-3xl font-bold text-white">מרכז הכניסה המאובטח ל-LexiAI</h1>
            <p className="text-sm leading-relaxed text-slate-200/80">
              התחברות באמצעות טוקן חתום המאפשר גישה לכלי הבינה, ניהול המסמכים ויכולות הציות של הפלטפורמה. אין לשתף את
              פרטי ההתחברות ללא אישור מנהלי המערכת.
            </p>
          </div>
          <div className="hidden gap-4 rounded-3xl border border-white/10 bg-white/10 p-6 text-xs text-slate-200/80 sm:grid">
            <p className="text-sm font-semibold text-white">מסלול מהיר להתחלה:</p>
            <ol className="space-y-2">
              <li>1. בקשו ממנהל המערכת שלכם טוקן התחברות ייעודי.</li>
              <li>2. הגדירו Tenant ID עבור הארגון שלכם בעת ההתחברות.</li>
              <li>3. היכנסו למרכז השליטה והתחילו לנתח תיקים ומסמכים.</li>
            </ol>
          </div>
          <Link
            href="/"
            className="inline-flex items-center justify-center rounded-full border border-white/20 px-5 py-3 text-sm font-semibold text-white transition hover:border-white/40 hover:bg-white/10"
          >
            חזרה לעמוד הבית ↗
          </Link>
        </section>
        <section className="rounded-3xl border border-white/15 bg-slate-950/70 p-8 shadow-inner">
          <form onSubmit={handleSubmit} className="space-y-5 text-right">
            <div>
              <label className="text-xs font-semibold text-blue-200/70" htmlFor="tenant">
                מזהה דייר
              </label>
              <input
                id="tenant"
                className="mt-2 w-full rounded-2xl border border-white/20 bg-white/5 px-4 py-3 text-sm text-white placeholder:text-slate-400 focus:border-blue-300 focus:outline-none"
                value={tenantId}
                onChange={(event) => setTenantId(event.target.value)}
                placeholder="לדוגמה: firm-123"
                required
              />
            </div>
            <div>
              <label className="text-xs font-semibold text-blue-200/70" htmlFor="email">
                דואר אלקטרוני
              </label>
              <input
                id="email"
                className="mt-2 w-full rounded-2xl border border-white/20 bg-white/5 px-4 py-3 text-sm text-white placeholder:text-slate-400 focus:border-blue-300 focus:outline-none"
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
                autoComplete="username"
              />
            </div>
            <div>
              <label className="text-xs font-semibold text-blue-200/70" htmlFor="password">
                סיסמה
              </label>
              <input
                id="password"
                className="mt-2 w-full rounded-2xl border border-white/20 bg-white/5 px-4 py-3 text-sm text-white placeholder:text-slate-400 focus:border-blue-300 focus:outline-none"
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
                autoComplete="current-password"
              />
            </div>
            {error ? <p className="text-sm text-red-300">{error}</p> : null}
            <button
              type="submit"
              className="w-full rounded-full bg-blue-500 px-6 py-3 text-sm font-semibold text-white shadow-[0_20px_60px_rgba(59,130,246,0.35)] transition hover:bg-blue-600"
              disabled={loading}
            >
              {loading ? "מתחבר..." : "כניסה למערכת"}
            </button>
          </form>
          <p className="mt-6 text-center text-[0.75rem] text-slate-400">
            עדיין אין לכם גישה? צרו קשר עם מנהל החשבון ב-LexiAI כדי להקים את סביבת הייצור שלכם.
          </p>
        </section>
      </div>
    </main>
  );
}
