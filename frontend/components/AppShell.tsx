"use client";

import Link from "next/link";
import { useRouter } from "next/router";
import { ReactNode, useMemo } from "react";

import { useAuth } from "../lib/auth";

const navigation = [
  {
    title: "כלי בינה משפטית",
    links: [
      { href: "/chat", label: "חדר פיקוד משפטי", description: "שוחח עם העוזר הקונטקסטואלי והפק תובנות בזמן אמת." },
      { href: "/predict", label: "חיזוי תיקים", description: "מודלי ML עם רציונל והמלצות אופרטיביות." },
      { href: "/witness", label: "חקירת עדים", description: "בנה תסריטי חקירה עם שאלות המשך ותיעדוף סיכונים." },
    ],
  },
  {
    title: "תפעול וציות",
    links: [
      { href: "/documents", label: "ניהול מסמכים", description: "גרסאות, רגולציה ותובנות ML למסמכים הקריטיים שלך." },
      { href: "/upload", label: "העלאות ותיעוד", description: "העלה חומרים חדשים וקבע מדיניות שימור ותיוג רגישות." },
      { href: "/compliance", label: "בקרת ציות", description: "בצע מעקב אחרי בקשות פרטיות, משימות ואירועים מבוקרים." },
    ],
  },
];

function classNames(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

type AppShellProps = {
  title: string;
  subtitle?: string;
  hero?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
};

export function AppShell({ title, subtitle, hero, actions, children }: AppShellProps) {
  const router = useRouter();
  const { session, logout } = useAuth();
  const activePath = router.pathname;

  const userSummary = useMemo(() => {
    if (!session) {
      return "אורח";
    }
    return `${session.userId} · ${session.tenantId}`;
  }, [session]);

  const handleLogout = async () => {
    logout();
    await router.replace("/login");
  };

  return (
    <div className="relative min-h-screen bg-slate-950 text-slate-100">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 right-1/2 h-[28rem] w-[28rem] translate-x-1/2 rounded-full bg-blue-500/30 blur-[120px]" />
        <div className="absolute bottom-[-6rem] left-[-4rem] h-[22rem] w-[22rem] rounded-full bg-purple-500/20 blur-[120px]" />
        <div className="absolute top-1/2 left-[10%] hidden h-64 w-64 -translate-y-1/2 rounded-full bg-cyan-500/20 blur-3xl lg:block" />
      </div>
      <div className="relative flex min-h-screen flex-row-reverse">
        <aside className="hidden w-[20rem] flex-shrink-0 flex-col border-l border-white/10 bg-slate-900/40 px-7 py-10 backdrop-blur-xl lg:flex">
          <Link href="/" className="flex items-center justify-between text-sm font-semibold text-blue-100">
            <span className="text-[0.7rem] font-bold uppercase tracking-[0.6em] text-blue-200">LexiAI</span>
            <span className="rounded-full bg-blue-500/20 px-3 py-1 text-[0.65rem] font-semibold text-blue-100">Platform</span>
          </Link>
          <div className="mt-6 rounded-3xl border border-white/10 bg-white/10 p-4 text-xs leading-relaxed text-blue-100/90 shadow-lg">
            <p className="font-semibold text-blue-50">מרכז השליטה המשפטי</p>
            <p className="mt-2 text-[0.7rem] text-blue-100/80">
              גישה לכלי הבינה המתקדמים, לתשתית הנתונים ולבקרות הציות של LexiAI בממשק אחד רציף.
            </p>
          </div>
          <nav className="mt-8 flex-1 space-y-8 overflow-y-auto pr-2 text-right">
            {navigation.map((section) => (
              <div key={section.title}>
                <p className="text-[0.65rem] font-semibold uppercase tracking-[0.3em] text-blue-200/70">{section.title}</p>
                <ul className="mt-3 space-y-2">
                  {section.links.map((link) => {
                    const isActive = activePath === link.href;
                    return (
                      <li key={link.href}>
                        <Link
                          href={link.href}
                          className={classNames(
                            "group block rounded-2xl border border-transparent px-4 py-3 transition", 
                            isActive
                              ? "border-blue-300/60 bg-blue-500/15 text-white shadow-[0_0_24px_rgba(59,130,246,0.25)]"
                              : "hover:border-white/20 hover:bg-white/10 text-slate-200",
                          )}
                        >
                          <p className="text-sm font-semibold">{link.label}</p>
                          <p className="mt-1 text-[0.7rem] text-slate-200/70">{link.description}</p>
                        </Link>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}
          </nav>
          <div className="mt-6 space-y-3 text-right text-xs text-slate-200/80">
            <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
              <p className="text-[0.65rem] font-semibold text-blue-200/80">משתמש מחובר</p>
              <p className="mt-1 text-[0.75rem] font-semibold text-white">{userSummary}</p>
            </div>
            <button
              type="button"
              onClick={handleLogout}
              className="w-full rounded-2xl border border-white/20 px-4 py-3 text-[0.75rem] font-semibold text-slate-200 transition hover:border-red-300/40 hover:bg-red-500/10 hover:text-red-200"
            >
              התנתקות מאובטחת
            </button>
          </div>
        </aside>
        <div className="flex min-h-screen flex-1 flex-col">
          <header className="border-b border-white/10 bg-slate-900/40 px-6 py-8 text-right backdrop-blur-xl">
            <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="text-[0.7rem] font-semibold uppercase tracking-[0.4em] text-blue-200/70">LexiAI Suite</p>
                <h1 className="mt-2 text-3xl font-bold text-white md:text-4xl">{title}</h1>
                {subtitle ? <p className="mt-3 max-w-3xl text-sm text-slate-200/80">{subtitle}</p> : null}
              </div>
              <div className="flex flex-col items-end gap-3 text-xs text-slate-200/80">
                <span className="rounded-full border border-white/20 px-4 py-2 text-[0.75rem] font-medium">
                  {userSummary}
                </span>
                {actions ? <div className="flex flex-wrap justify-end gap-2">{actions}</div> : null}
              </div>
            </div>
            {hero ? <div className="mt-8">{hero}</div> : null}
          </header>
          <main className="flex-1 px-6 py-10">
            <div className="space-y-10">{children}</div>
          </main>
        </div>
      </div>
    </div>
  );
}
