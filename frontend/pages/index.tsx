"use client";

import Link from "next/link";
import { useRouter } from "next/router";

import { useAuth } from "../lib/auth";

const featureHighlights = [
  {
    title: "תובנות מיידיות",
    description:
      "חוויית RAG שמחברת בין תיקי הלקוחות, מסמכי המשרד והקורפוס המשפטי של LexiAI כדי לענות בשפה טבעית עם רציונל שקוף.",
  },
  {
    title: "שליטה מלאה בתשתית",
    description:
      "בריכת נתונים רב-דיירית, גיבויים אוטומטיים ויומני פעילות מקיפים שמבטיחים שליטה מלאה בכל גרסה ובכל פעולה.",
  },
  {
    title: "ציות שנבנה-כברירת-מחדל",
    description:
      "בקשות פרטיות, מדיניות שימור וזרימות עבודה ארגוניות מרוכזות בקונסול יחיד עם קווי מדיניות מובנים מראש.",
  },
];

const capabilityGrid = [
  {
    eyebrow: "עיבוד מסמכים",
    title: "זיהוי התחייבויות, סיכונים ודדליינים",
    body: "המערכת מפיקה תמצית קונטקסטואלית, רציונל ML ומדדי ביטחון לכל מסמך שמועלה לפלטפורמה.",
  },
  {
    eyebrow: "אסטרטגיית עדות",
    title: "חדר מלחמה לחקירות עדים",
    body: "בנו תרחישי חקירה עם שאלות המשך, תיעדוף נושאים והצלבת אסמכתאות מתוך המסמכים של הלקוח והפסיקה.",
  },
  {
    eyebrow: "אינטגרציית LLM",
    title: "Claude כשותף אסטרטגי",
    body: "חיבור ישיר למודל של Anthropic עם מנגנוני נפילה, ניטור ותיעוד כדי להבטיח תוצאות קונסיסטנטיות ומבוקרות.",
  },
  {
    eyebrow: "בקרת ציות",
    title: "תיקוף בקשות ויומני ביקורת",
    body: "תעדפו בקשות פרטיות, עקבו אחרי משימות והפיקו דוחות מוכנים לביקורת – הכל ממסך אחד.",
  },
];

const launchSteps = [
  "התחברו עם טוקן מאובטח או תהליך Single Sign-On ייעודי.",
  "קבעו את פרטי הדייר, מסד הנתונים וגיבויי הייצור ישירות מהקונסול.",
  "העלו מסמכים, הגדירו מדיניות והתחילו לקבל תובנות בזמן אמת בכל מודולי LexiAI.",
];

export default function Home() {
  const router = useRouter();
  const { session, logout } = useAuth();

  const handleLogout = async () => {
    logout();
    await router.replace("/login");
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute top-[-12rem] right-[-12rem] h-[28rem] w-[28rem] rounded-full bg-blue-500/20 blur-[160px]" />
        <div className="absolute bottom-[-10rem] left-[-16rem] h-[32rem] w-[32rem] rounded-full bg-purple-500/25 blur-[160px]" />
      </div>
      <div className="relative mx-auto flex min-h-screen max-w-6xl flex-col px-6 py-12 text-right">
        <header className="flex flex-col gap-8">
          <div className="flex flex-col items-end justify-between gap-6 sm:flex-row sm:items-center">
            <div className="flex items-center gap-3 text-xs font-semibold text-blue-200">
              <span className="rounded-full border border-blue-300/40 px-3 py-1 uppercase tracking-[0.4em]">LexiAI</span>
              <span className="hidden sm:inline text-slate-200/70">Platform Edition</span>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-200/80">
              {session ? (
                <>
                  <span className="rounded-full border border-white/20 px-3 py-1 font-semibold">
                    {session.userId} · {session.tenantId}
                  </span>
                  <button
                    type="button"
                    onClick={handleLogout}
                    className="rounded-full border border-white/20 px-4 py-2 font-semibold text-slate-200 transition hover:border-red-200/60 hover:text-red-100"
                  >
                    התנתקות
                  </button>
                </>
              ) : (
                <Link
                  href="/login"
                  className="rounded-full border border-blue-300/60 px-4 py-2 font-semibold text-blue-200 transition hover:bg-blue-500/20"
                >
                  כניסה ללקוחות
                </Link>
              )}
            </div>
          </div>
          <div className="space-y-6">
            <h1 className="text-4xl font-black leading-tight text-white sm:text-5xl">
              בנו משרד משפטי דיגיטלי עם חוויית AI מקצה לקצה
            </h1>
            <p className="max-w-3xl text-lg text-slate-200/80">
              LexiAI מאחד את כל התיקים, המסמכים וזרימות הציות למרכז שליטה אחד: צ'אט משפטי קונטקסטואלי, חיזוי תיקים
              מוסבר, חקירת עדים מונעת נתונים ומעקב רגולטורי בזמן אמת.
            </p>
            <div className="flex flex-wrap justify-end gap-3 text-sm">
              <Link
                href="/chat"
                className="rounded-full bg-blue-500 px-6 py-3 font-semibold text-white shadow-[0_20px_60px_rgba(59,130,246,0.35)] transition hover:bg-blue-600"
              >
                התחל בשולחן הפיקוד
              </Link>
              <Link
                href="/documents"
                className="rounded-full border border-white/30 px-6 py-3 font-semibold text-white transition hover:border-white/50 hover:bg-white/10"
              >
                סייר במסמכים
              </Link>
            </div>
          </div>
        </header>

        <section className="mt-16 grid gap-6 lg:grid-cols-[2fr_3fr]">
          <div className="glass-panel space-y-6">
            <p className="text-sm font-semibold uppercase tracking-[0.3em] text-blue-200/70">מה מקבלים</p>
            <h2 className="text-2xl font-bold text-white">חוויית משתמש מודרנית עם תפעול ארגוני מלא</h2>
            <p className="text-sm text-slate-200/80">
              הפלטפורמה מעוצבת למשרדי עורכי דין וצוותי ליטיגציה הדורשים בינה אמינה, אבטחת מידע ותהליכים ברורים. כל מודול
              מתחבר לנתונים הארגוניים ומציג מדדי הצלחה ברורים.
            </p>
            <ol className="space-y-3 text-sm text-slate-200/90">
              {launchSteps.map((item, index) => (
                <li key={item} className="flex items-start gap-3">
                  <span className="mt-1 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-blue-500/30 text-sm font-bold text-blue-100">
                    {index + 1}
                  </span>
                  <span className="leading-relaxed">{item}</span>
                </li>
              ))}
            </ol>
          </div>
          <div className="grid gap-6 sm:grid-cols-2">
            {featureHighlights.map((feature) => (
              <div key={feature.title} className="glass-panel h-full space-y-3 border-white/5">
                <h3 className="text-lg font-semibold text-white">{feature.title}</h3>
                <p className="text-sm leading-relaxed text-slate-200/80">{feature.description}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="mt-16 rounded-3xl border border-white/10 bg-white/5 p-8 shadow-2xl">
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-blue-200/70">יכולות הליבה</p>
          <div className="mt-6 grid gap-6 md:grid-cols-2">
            {capabilityGrid.map((capability) => (
              <div key={capability.title} className="rounded-3xl border border-white/10 bg-slate-900/60 p-6">
                <p className="text-xs font-semibold text-blue-200/70">{capability.eyebrow}</p>
                <h3 className="mt-2 text-xl font-semibold text-white">{capability.title}</h3>
                <p className="mt-3 text-sm leading-relaxed text-slate-200/80">{capability.body}</p>
                <Link
                  href="/login"
                  className="mt-6 inline-flex items-center text-sm font-semibold text-blue-200 hover:text-blue-100"
                >
                  למדו עוד במרכז השליטה ↗
                </Link>
              </div>
            ))}
          </div>
        </section>

        <section className="mt-16 grid gap-6 lg:grid-cols-[3fr_2fr]">
          <div className="glass-panel space-y-4">
            <p className="text-sm font-semibold uppercase tracking-[0.3em] text-blue-200/60">מדדי השפעה</p>
            <div className="grid gap-4 sm:grid-cols-3">
              {["24/7", "100%", "7 ימים"].map((metric, index) => (
                <div key={metric} className="rounded-2xl border border-white/10 bg-white/5 p-4 text-right">
                  <p className="text-2xl font-bold text-white">{metric}</p>
                  <p className="mt-1 text-xs text-slate-200/70">
                    {index === 0 && "תמיכת בינה רציפה"}
                    {index === 1 && "כיסוי מלא ליכולות LexiAI"}
                    {index === 2 && "זמן ממוצע להטמעה"}
                  </p>
                </div>
              ))}
            </div>
            <p className="text-sm text-slate-200/70">
              נתונים אלה מבוססים על סבבי הפיילוט האחרונים שלנו ומדגימים כיצד צוותים משפטיים הפכו את LexiAI לחלק מהפעילות
              היומיומית שלהם.
            </p>
          </div>
          <div className="glass-panel space-y-4">
            <h3 className="text-xl font-semibold text-white">הזמנה להדגמה חיה</h3>
            <p className="text-sm text-slate-200/80">
              הצטרפו לסבב הדגמות שבועי עם מומחי המוצר, למדו כיצד LexiAI משתלב בארגון שלכם וקבלו תכנית פריסה מותאמת.
            </p>
            <Link
              href="/login"
              className="inline-flex items-center justify-center rounded-full bg-blue-500 px-6 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-blue-600"
            >
              בקש/י גישה מוקדמת
            </Link>
          </div>
        </section>

        <footer className="mt-24 flex flex-col items-center gap-4 border-t border-white/10 pt-8 text-center text-xs text-slate-300/70">
          <span>© {new Date().getFullYear()} LexiAI. כל הזכויות שמורות.</span>
          <div className="flex flex-wrap justify-center gap-4">
            <Link href="/login">כניסה</Link>
            <Link href="/documents">מסמכים</Link>
            <Link href="/compliance">ציות</Link>
          </div>
        </footer>
      </div>
    </div>
  );
}
